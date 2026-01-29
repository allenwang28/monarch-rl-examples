# CLAUDE.md

## Project Overview

This is a pedagogical repository demonstrating Monarch applied to RL infrastructure challenges. Target audience is technical folks who know PyTorch but not RL infra or distributed systems deeply.

## Key Constraints

**Dependencies:** Only `torch`, `transformers` (HuggingFace), and `monarch`. Keep the installation footprint small.

**No imports from:**
- Forge (we're building patterns from scratch)
- TorchTitan (too heavy)
- TorchStore (we'll show the concepts, not use the library)
- vLLM (keep it simple with HuggingFace)

**Can reference for ideas:**
- `dtensor_transfer/` - steal patterns from here
- `forge/src/forge/` - for understanding service patterns
- `monarch/` - the actual library we're using

## Style Guidelines

- **Tone:** Straightforward, technical. No dramatic language ("the villain", "the star", etc.)
- **Code:** Clear, commented, runnable. Prioritize readability over cleverness.
- **Docs:** Concept-first with code as illustration. Explain the "why" not just the "how".
- **Diagrams:** ASCII is fine. Keep them in the docs or in `diagrams/` folder.

## Repository Structure

```
.claude/        # Project context and planning docs
docs/           # Markdown documentation (concept docs)
examples/       # Runnable Python examples (01-07)
diagrams/       # ASCII diagrams referenced by docs
viz/            # Optional visualizations
```

See `.claude/PLAN.md` for the full implementation plan.

## Key Concepts to Understand

1. **Control plane vs Data plane:** Actor messages for coordination (small), RDMA for bulk data (fast)
2. **CPU staging:** Copy weights to CPU, transfer via RDMA, copy to destination GPU - allows overlap
3. **Service pattern:** Wrap multiple actor replicas, route to healthy ones, retry on failure
4. **Policy version:** Track which policy generated each trajectory for staleness management

## Running Examples

Examples are designed to run on a machine with GPUs. The final example (07_async_rl_train.py) requires at least 2 GPUs.

```bash
# Most examples
python examples/01_hello_actor.py

# The full RL training example
python examples/07_async_rl_train.py
```

## The Training Task

We're using Qwen 2.5 0.7B for the final example. The specific task (repeat, count, or pattern-following) is TBD based on testing which one shows learning best.

Goal: Print trajectories during training and show reward improvement over time.

## What NOT to Do

- Don't add complex error handling - keep examples focused
- Don't optimize for production - optimize for clarity
- Don't add multi-host support - out of scope for runnable examples
- Don't use process groups for weight sync - that's what we're showing an alternative to

## Related Files in the Monorepo

- `/home/allencwang/forge-dev/monarch/` - The Monarch library
- `/home/allencwang/forge-dev/smol_projects/dtensor_transfer/` - Reference for RDMA patterns
- `/home/allencwang/forge-dev/forge/src/forge/controller/service/` - Reference for service patterns
- `/home/allencwang/forge-dev/forge/apps/grpo/` - Reference for full RL implementation
