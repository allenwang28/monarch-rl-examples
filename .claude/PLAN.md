# Monarch RL Examples: Implementation Plan

## Overview

**Goal:** Build a pedagogical repo that demonstrates Monarch applied to RL infrastructure challenges. Target audience is GPU Mode - technical folks who know PyTorch but not RL infra.

**Core Messages:**
1. Monarch makes distributed programming feel local
2. RDMA doesn't have to be intimidating - think "magic pointer"
3. Weight sync without process groups is more elegant
4. Async RL is just two loops + a buffer

---

## Problems We're Addressing

**Process groups for weight sync:**
- Blocking synchronization
- Tightly coupled to world size
- Hard to have different sharding between components

**Traditional SPMD:**
- Rank-specific code paths
- Implicit synchronization
- Hard to reason about

---

## Repository Structure

```
monarch-rl-examples/
├── .claude/
│   ├── CLAUDE.md                   # Project context for Claude
│   └── PLAN.md                     # This file
│
├── docs/
│   ├── 00_overview.md              # What we're building, why it matters
│   ├── 01_monarch_primer.md        # Quick refresher (assumes someone else covers depth)
│   ├── 02_weight_sync.md           # Main focus - weight sync patterns
│   ├── 03_services.md              # Fault-tolerant routing from scratch
│   ├── 04_resharding.md            # DTensor as unified language
│   └── 05_async_rl.md              # The full loop explained
│
├── examples/
│   │   # Part 1: Monarch Fundamentals
│   ├── 01_hello_actor.py           # Minimal actor example
│   ├── 02_actor_mesh.py            # Mesh + adverbs (call, broadcast, etc)
│   ├── 03_rdma_transfer.py         # Control plane vs data plane
│   ├── 04_controllers.py           # get_or_spawn_controller, service discovery
│   ├── 05_supervision.py           # __supervise__ hook, fault detection
│   │   # Part 2: RL Infrastructure
│   ├── 06_simple_service.py        # Route + round robin from scratch
│   ├── 07_weight_sync.py           # CPU staging + RDMA pattern
│   ├── 08_dtensor_reshard.py       # Steal from dtensor_transfer
│   └── 09_async_rl_train.py        # Real training with Qwen 0.7B
│
├── diagrams/                       # ASCII diagrams for docs
│   ├── blocking_pg.md              # Traditional PG approach
│   ├── rdma_offload.md             # CPU staging + RDMA approach
│   └── async_rl_loop.md            # Two loops + buffer
│
└── viz/                            # Optional: interactive visualization
    └── resource_utilization/       # Slider thing for async RL demo
```

---

## Documents Outline

### 00_overview.md
- What is async RL at a high level?
- Two parallel loops: generation feeds training, training feeds generation
- The infrastructure challenges we'll solve
- What we're building toward

### 01_monarch_primer.md
- Assumes someone else covers Monarch in depth
- Quick refresher on: Actor, ProcMesh, ActorMesh, endpoint, adverbs
- Enough to follow along, not comprehensive

### 02_weight_sync.md

**Structure:**
1. The Problem
   - Trainer updates weights every step
   - Generators need fresh weights
   - How fast? How fresh? The latency budget.

2. The Traditional Approach
   - Process groups (NCCL all_gather / broadcast)
   - Blocking: both sides wait
   - Tightly coupled: same world size, same PG

3. The Monarch Approach
   - Control plane: actor messages (tiny, fast)
   - Data plane: RDMA (bulk, direct)
   - CPU staging: offload to CPU, transfer async, copy back

4. The Mental Model
   - RDMABuffer = "magic pointer"
   - Pass the handle (tiny), receiver pulls data (fast)

5. The Key Diagram
   - Side-by-side: Blocking PG vs RDMA offload
   - Timeline showing overlap

6. Design Decisions
   - Why not serialize through actor messages?
   - Why RDMA over collectives?
   - Why CPU staging?

7. Code Walkthrough
   - Link to example 04_weight_sync.py

### 03_services.md
- The problem: call any healthy replica, not a specific actor
- Building the abstraction from scratch
- Route + round robin
- Brief mention of: load balancing, sticky sessions, health monitoring (concepts, not full impl)

### 04_resharding.md
- The problem: trainer sharding ≠ generator sharding
- Traditional: full gather then transfer
- Ideal: compute overlaps, transfer only what's needed
- DTensor as the unified language
- Preview: this is where torchtitan RL is heading
- Steal concepts from dtensor_transfer

### 05_async_rl.md
- The two loops diagram
- Generator → Buffer flow
- Buffer → Trainer → Generator flow
- How they run in parallel
- The buffer as the only sync point

---

## Examples Outline

### Part 1: Monarch Fundamentals

### 01_hello_actor.py
**Concepts:** Actor, endpoint, this_proc, spawn, call_one

```python
# ~20 lines
# Spawn an actor, call an endpoint, get a result
# Print which GPU it's on
```

### 02_actor_mesh.py
**Concepts:** ProcMesh, ActorMesh, call, broadcast, choose

```python
# ~40 lines
# Spawn 4 actors across 4 processes
# Demo each adverb: call(), broadcast(), call_one(), choose()
# Show that state is local to each actor
```

### 03_rdma_transfer.py
**Concepts:** RDMABuffer, read_into, control vs data plane

```python
# ~60 lines
# Actor A has a big tensor
# Actor A exposes RDMABuffer handle
# Actor B gets handle (small message)
# Actor B calls read_into (bulk transfer)
# Compare: passing tensor directly vs RDMA
```

### 04_controllers.py
**Concepts:** get_or_spawn_controller, singleton actors, service discovery

```python
# ~60 lines
# Multiple actors finding the same controller by name
# First caller creates, subsequent callers get existing
# No need to pass references around
# Example: shared metrics aggregator or config store
```

### 05_supervision.py
**Concepts:** __supervise__ hook, fault detection, parent-child relationships

```python
# ~60 lines
# Parent actor spawns child actors
# Child actor fails (crashes or raises)
# Parent's __supervise__ is called with failure info
# Show how to log, recover, or propagate the failure
```

### Part 2: RL Infrastructure

### 06_simple_service.py
**Concepts:** Service abstraction, route, round robin, retry on failure

```python
# ~80 lines
# Multiple replica actors (e.g., 4 generators)
# Service wrapper that:
#   - Tracks healthy replicas
#   - Routes requests round-robin
#   - Retries on failure
# Demo: call service.route(request) multiple times
```

### 07_weight_sync.py
**Concepts:** CPU staging, async transfer, weight sync pattern

```python
# ~100 lines
# Trainer actor with model weights
# Generator actor that needs weights
# Pattern:
#   1. Trainer copies weights to CPU staging buffer
#   2. Trainer exposes RDMABuffer handle
#   3. Generator pulls via RDMA (non-blocking for trainer)
#   4. Generator copies to GPU
# Show timeline of operations
```

### 08_dtensor_reshard.py
**Concepts:** DTensor, different shardings, overlap transfer

```python
# ~100 lines
# Steal from dtensor_transfer
# Show: tensor sharded as Shard(0) on mesh A
# Transfer to: tensor sharded as Shard(1) on mesh B
# Show overlap computation concept
```

### 09_async_rl_train.py
**Concepts:** Full async RL loop, real training

```python
# ~200 lines
# Components:
#   - Generator (Qwen 2.5 0.7B via HuggingFace)
#   - Trainer (simple policy gradient)
#   - Replay buffer (in-memory, policy version tracking)
#   - Weight sync (using pattern from 07)
#
# Two async loops:
#   - continuous_rollouts(): generate → score → buffer
#   - continuous_training(): buffer → train → sync weights
#
# Prints:
#   - Each trajectory (prompt, response, reward)
#   - Running average reward
#   - Policy version
#
# Task: Multi-turn (TBD - test repeat vs count)
```

---

## The Training Task

**Options to test:**

### Option A: "Repeat after me" (multi-turn)
```
Turn 1: "I'll say a word, you repeat it. Ready?"
Turn 2: "banana"
Expected: "banana"
Reward: 1 if exact, 0 otherwise

Turn 3: "telescope"
Expected: "telescope"
...
```

### Option B: "Learn to count"
```
Prompt: "Count to 5, one number per line"
Expected: "1\n2\n3\n4\n5"
Reward: fraction of correct numbers in sequence
```

### Option C: "Follow the pattern"
```
Turn 1: "When I say X, you say Y. When I say A, you say B."
Turn 2: "X"
Expected: "Y"
Turn 3: "A"
Expected: "B"
```

**Action item:** Test all three on Qwen 2.5 0.7B to see which one it's bad at (room for learning) but not hopeless.

---

## Key Diagrams

### Diagram 1: Blocking PG vs RDMA Offload

```
BLOCKING PROCESS GROUPS:
========================

TRAINER:    ├── train ──┼══ BLOCKED ══┼── train ──┤
                        │             │
                        │  all_gather │
                        │             │
GENERATOR:  ├── gen ────┼══ BLOCKED ══┼── gen ────┤

Timeline: ════════════════════════════════════════▶
          Both sides wait. GPU idle during sync.


RDMA WITH CPU OFFLOAD:
======================

TRAINER:    ├── train ──┼── train ──┼── train ──┤
                  │ copy to CPU (fast)
                  ▼
CPU STAGING:      └──── RDMA transfer ────┘
                                          │ copy to GPU
                                          ▼
GENERATOR:  ├── gen ────┼── gen ────┼── gen (new) ──┤

Timeline: ════════════════════════════════════════▶
          Everything overlaps. GPUs stay busy.
```

### Diagram 2: Control Plane vs Data Plane

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  CONTROL PLANE                    DATA PLANE            │
│  (Actor Messages)                 (RDMA)                │
│                                                         │
│  ┌─────────────────┐              ┌─────────────────┐   │
│  │ "Here's my      │              │                 │   │
│  │  weights handle"│              │  10 GB tensor   │   │
│  │                 │              │  transfer       │   │
│  │  ~100 bytes     │              │                 │   │
│  └────────┬────────┘              └────────▲────────┘   │
│           │                                │            │
│           ▼                                │            │
│  trainer.get_handle()          handle.read_into(buf)   │
│  → instant                     → triggers bulk xfer    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Diagram 3: Async RL Loop

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│          ┌─────────────┐                                │
│          │  GENERATOR  │                                │
│          │  (Qwen 0.7B)│                                │
│          └──────┬──────┘                                │
│                 │ episodes                              │
│                 ▼                                       │
│          ┌─────────────┐                                │
│          │   BUFFER    │◀─── sample ───┐               │
│          │(policy ver) │                │               │
│          └──────┬──────┘                │               │
│                 │ batches               │               │
│                 ▼                       │               │
│          ┌─────────────┐                │               │
│          │   TRAINER   │────weights────▶│               │
│          └─────────────┘                                │
│                 │                       │               │
│                 └───── weights ─────────┘               │
│                        (RDMA)                           │
│                                                         │
│  Loop 1: generate → score → add to buffer (continuous)  │
│  Loop 2: sample → train → push weights (continuous)     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Visualization Ideas

### Resource Utilization Slider

For the podcast, it would be cool to have an interactive visualization that shows:

- X-axis: time
- Y-axis: resource utilization (GPU %)
- Slider: "sync ratio" from fully-sync to fully-async

As you slide from sync → async:
- Blocking gaps shrink
- Overall utilization goes up
- Latency vs throughput trade-off becomes visible

**Implementation options:**
1. Simple: Matplotlib animation saved as GIF
2. Medium: Streamlit app with slider
3. Fancy: Web-based with D3.js

**Question:** How much time to invest here? Could be a nice-to-have.

---

## What to Steal from dtensor_transfer

| Source | What to take | For |
|--------|--------------|-----|
| `remote_tensor.py` | RDMABuffer wrapping pattern | 03_rdma_transfer.py |
| `remote_tensor.py` | IPC vs RDMA transport selection | 07_weight_sync.py |
| `gather.py` | Basic gather approach | 08_dtensor_reshard.py |
| `routed.py` | Overlap computation concept | 08_dtensor_reshard.py |
| `README.md` | Performance numbers, diagrams | docs/04_resharding.md |

---

## Priority Order

**P0 (Must have for podcast):**
1. 02_weight_sync.md + 07_weight_sync.py - Weight sync patterns
2. 03_rdma_transfer.py - Sets up the mental model
3. 09_async_rl_train.py - Full training example
4. Key diagrams

**P1 (Should have):**
5. 06_simple_service.py - Service abstraction
6. 04_controllers.py - Service discovery pattern
7. 05_supervision.py - Fault detection basics
8. 08_dtensor_reshard.py - Future direction
9. 00_overview.md, 05_async_rl.md - Context docs

**P2 (Nice to have):**
10. Resource utilization visualization
11. 01_hello_actor.py, 02_actor_mesh.py - Basics (may be covered by others)

---

## Open Questions

1. **Training task:** Test repeat vs count vs pattern on Qwen 2.5 0.7B. Which shows learning best?

2. **Visualization:** How much time to invest in the resource utilization slider? Matplotlib GIF vs Streamlit vs skip?

3. **Services depth:** Just route + round robin, or also show health monitoring concept?

4. **Multi-host:** Explicitly out of scope for runnable examples? (Mention in docs only?)

5. **Model size:** 0.7B confirmed? Or test with 0.5B for faster iteration?

---

## Success Criteria

- [ ] Someone can clone the repo and run examples 01-07
- [ ] The weight sync diagram clearly shows the RDMA advantage
- [ ] 07_async_rl_train.py prints trajectories and shows reward improvement
- [ ] Docs explain the "why" not just the "how"
- [ ] Can walk through in ~3 hours with good pacing

---

## Next Steps

1. Test training tasks on Qwen 2.5 0.7B
2. Implement 03_rdma_transfer.py (foundation)
3. Implement 04_weight_sync.py (main focus)
4. Write 02_weight_sync.md with diagrams
5. Implement 07_async_rl_train.py
6. Fill in remaining examples and docs
7. Optional: resource utilization viz
