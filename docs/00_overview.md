# Overview: What We're Building

## Why RL for LLMs?

Reinforcement learning has become a core part of the LLM pipeline:

- **RLHF** - How we got instruction-following models. The model learns to produce outputs humans prefer.
- **Reasoning** - Models like o1 and DeepSeek-R1 use RL to learn step-by-step thinking. Generate many chains of thought, reward the ones that reach correct answers.
- **Agentic training** - Teaching models to use tools, browse the web, write code that runs. The reward comes from whether the action worked.

The common thread: we're aligning models to behaviors we care about, not just predicting the next token.

## What Makes RL Different?

In supervised learning, you have a fixed dataset. The training loop is:

```
for batch in dataset:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

In RL, the model generates its own training data. The loop becomes:

```
while training:
    trajectories = model.generate(prompts)      # <-- new: model creates data
    rewards = evaluate(trajectories)             # <-- new: score what it made
    loss = rl_loss(trajectories, rewards)
    loss.backward()
    optimizer.step()
```

That `generate` step is expensive. For a 7B model generating long responses, it can take 10x longer than the training step. If you do this synchronously, your training GPUs sit idle while waiting for generation.

## The Infrastructure Challenge

Most discussions of RL focus on algorithms (PPO, GRPO, DPO). But at scale, the infrastructure matters just as much:

- **How do you keep GPUs busy?** Generation and training should run in parallel.
- **How do you move weights?** The generator needs fresh weights from the trainer.
- **How do you handle failures?** With many generators running, some will crash.

This is the stuff that's often tribal knowledge inside labs. We're going to build it from scratch.

## What We're Building

We're going to build a minimal async RL training system using Monarch and PyTorch. By the end, you'll understand the infrastructure challenges and see one clean way to solve them.

## What is Async RL?

At its core, RL training has two jobs:

1. **Generate experiences** - Run the model, collect trajectories, score them
2. **Learn from experiences** - Update the model weights based on what worked

In synchronous RL, these happen in lockstep: generate a batch, train on it, repeat. Simple, but GPUs sit idle waiting for each other.

```
SYNCHRONOUS RL
==============

Time ──────────────────────────────────────────────────────────▶

Generator:  ├── gen ──┤         ├── gen ──┤         ├── gen ──┤
                      │         │         │         │
                      ▼         │         ▼         │
Trainer:              ├─ train ─┤         ├─ train ─┤
                                │                   │
                           (idle)              (idle)

One thing happens at a time. When generating, trainer waits.
When training, generator waits.
```

In asynchronous RL, these run as independent loops:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   GENERATION LOOP              TRAINING LOOP            │
│                                                         │
│   ┌───────────┐                ┌───────────┐            │
│   │ Generator │──episodes──▶   │  Buffer   │            │
│   └─────▲─────┘                └─────┬─────┘            │
│         │                            │                  │
│         │                       batches                 │
│         │                            │                  │
│         │                      ┌─────▼─────┐            │
│         └──── weights ─────────│  Trainer  │            │
│                                └───────────┘            │
│                                                         │
│   Loop 1: generate → score → add to buffer              │
│   Loop 2: sample from buffer → train → push weights     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

The buffer decouples them. Generation keeps generating. Training keeps training. Weights flow from trainer back to generator so it stays (mostly) on-policy.

```
ASYNCHRONOUS RL (timeline view)
===============================

Time ──────────────────────────────────────────────────────────▶

Generator:  ├── gen ──┼── gen ──┼── gen ──┼── gen ──┼── gen ──┤
                 │         │         │
                 ▼         ▼         ▼
Buffer:     ════════════════════════════════════════════════════
                      │         │         │
                      ▼         ▼         ▼
Trainer:    ├─ train ─┼─ train ─┼─ train ─┼─ train ─┼─ train ─┤

Everything runs in parallel. No waiting.
```

**But there's a catch.**

## The On-Policy Trade-off

Here's the tension: RL works best when the model learns from data it *just* generated. The current policy's attempts are more relevant than an older policy's attempts.

In async RL, the generator is always a few steps behind the trainer:

```
Trainer:     v1 ─────▶ v2 ─────▶ v3 ─────▶ v4
Generator:      └─ v1 ──┼─ v1 ──┼─ v2 ──┤
                          ↓        ↓
                    (slightly stale data)
```

This means we're training on slightly off-policy data. Is that okay?

- **Small gap (1-2 versions)**: Usually fine. The policy hasn't changed much.
- **Large gap**: Problematic. We'll need to track and drop stale data.

The trade-off is worth it: ~2x throughput for a small staleness cost. But it's not free.

We'll dig deeper into managing staleness in [05_async_rl.md](05_async_rl.md).

## Why Monarch?

Async RL sounds great, but how do you actually build it? You need:

- Multiple processes (generators, trainer, buffer) running independently
- Communication between them (weights, trajectories)
- Fault handling when things crash

The challenge with `torch.distributed` is that it's designed for SPMD—everyone runs the same code, you check your rank to decide what to do. That works for training, but async RL has different roles (generator, trainer, buffer) doing different things.

Monarch gives you the actor model with PyTorch-native primitives:

```python
from monarch.actor import Actor, endpoint, this_host

class Generator(Actor):
    @endpoint
    async def generate(self, prompt: str) -> str:
        return self.model.generate(prompt)

# Spawn 4 generators across 4 GPUs
procs = this_host().spawn_procs(per_host={"gpus": 4})
generators = procs.spawn("generators", Generator)

# Call all of them
responses = await generators.generate.call(prompt="Hello")
```

That's it. No rank checks. No process group setup. Monarch handles the distribution.

**Key Monarch concepts we'll use:**

| Concept | What it is |
|---------|------------|
| **Actor** | Isolated state + endpoints. The basic unit. |
| **ProcMesh** | A grid of processes mapped to hardware (GPUs). |
| **ActorMesh** | Actors spawned across a ProcMesh. |
| **Endpoint** | A method you can call remotely. |
| **RDMABuffer** | A handle for fast bulk data transfer. |

The mental model: write Python that looks local, Monarch makes it distributed.

## Why is This Hard?

Three infrastructure challenges:

### 1. Weight Synchronization

The trainer updates weights every step. The generator needs those weights to stay on-policy. How do you move 10GB of weights quickly and frequently without blocking either side?

Traditional approach: Process groups (NCCL all-gather). Both sides block during transfer.

What we'll show: RDMA with CPU staging. Trainer keeps training while weights transfer in the background.

### 2. Fault Tolerance

With multiple generators running, some will fail. The system should keep going with the healthy ones and recover the failed ones.

Traditional approach: Restart the whole job.

What we'll show: A service abstraction that routes to healthy replicas and recovers failed ones automatically.

### 3. Different Parallelism Strategies

The trainer might use FSDP across 4 GPUs (row-sharded weights). The generator might use tensor parallelism across 2 GPUs (column-sharded weights). How do you transfer weights between them?

Traditional approach: Gather everything, then scatter.

What we'll show: DTensor-aware transfer that computes overlaps and only moves what's needed.

## What We'll Build

Seven examples, building up:

| Example | What it shows |
|---------|---------------|
| 01 | Hello Actor - the basic building block |
| 02 | Actor Mesh - running on multiple GPUs |
| 03 | RDMA Transfer - control plane vs data plane |
| 04 | Weight Sync - the CPU staging pattern |
| 05 | Simple Service - routing to healthy replicas |
| 06 | DTensor Reshard - different shardings |
| 07 | Async RL Train - putting it all together |

By example 07, you'll have a working async RL system training a real model, printing trajectories, and showing reward improvement.

## Prerequisites

We assume you know:
- PyTorch basics (tensors, models, training loops)
- Python async/await
- What GPUs are

We'll explain:
- The actor model
- RDMA and why it matters
- How to think about distributed RL

## Let's Go

Start with [01_monarch_primer.md](01_monarch_primer.md) for a quick refresher on Monarch, or jump straight to the examples if you want to see code.
