# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torchmonarch",
# ]
# ///
"""
02: Actor Mesh

Spawn actors across multiple processes and see how to communicate with them.

Run with: uv run examples/02_actor_mesh.py

Key concepts:
- ProcMesh: A grid of processes mapped to hardware
- ActorMesh: Actors spawned across a ProcMesh
- ValueMesh: The result of calling endpoints on a mesh (holds values per rank)
- Adverbs: Different ways to call endpoints on a mesh
  - .call(): Call all actors, get all results as ValueMesh
  - .broadcast(): Call all actors with same args
  - .call_one(): Call a single actor (use after .slice() to pick which one)
  - .choose(): Call one actor (system picks)
- .slice(): Select a subset of actors by coordinate
- .get(): Block and wait for the result (sync API)

This example shows that each actor has its own local state.
"""

from monarch.actor import Actor, endpoint, current_rank, this_host, ValueMesh


class Counter(Actor):
    """An actor that counts how many times it's been called."""

    def __init__(self):
        self.count = 0
        self.rank = current_rank()

    @endpoint
    def increment(self) -> dict:
        """Increment the count and return info about this actor."""
        self.count += 1
        return {"rank": self.rank, "count": self.count}

    @endpoint
    def get_count(self) -> int:
        """Return the current count."""
        return self.count

    @endpoint
    def add(self, value: int) -> int:
        """Add a value to the count."""
        self.count += value
        return self.count


def main():
    n_procs = 4
    print(f"Spawning {n_procs} processes...")

    procs = this_host().spawn_procs(per_host={"procs": n_procs})
    counters = procs.spawn("counters", Counter)

    print("\n--- call(): Call all actors ---")
    # call().get() returns a ValueMesh containing results from all actors
    results: ValueMesh[dict] = counters.increment.call().get()
    # Use .items() to iterate with coordinates, or .values() for just values
    for point, r in results.items():
        print(f"  Point {point}: count = {r['count']}")

    print("\n--- Calling again to show state persists ---")
    results = counters.increment.call().get()
    for r in results.values():
        print(f"  Rank {r['rank']}: count = {r['count']}")

    print("\n--- slice() + call_one(): Call a specific actor ---")
    # slice() selects actors by coordinate, then call_one() calls the single actor
    result = counters.slice(procs=0).increment.call_one().get()
    print(f"  Rank {result['rank']}: count = {result['count']}")

    print("\n--- Get all counts to see the difference ---")
    counts = counters.get_count.call().get()
    for i, c in enumerate(counts.values()):
        print(f"  Actor {i}: count = {c}")

    print("\n--- choose(): Let system pick one actor ---")
    # choose() picks one actor (useful for load balancing)
    result = counters.increment.choose().get()
    print(f"  Rank {result['rank']}: count = {result['count']}")

    print("\n--- broadcast(): Fire-and-forget to all actors ---")
    # broadcast() sends to all actors but doesn't wait for results
    print(f"  Counts before: {list(counters.get_count.call().get().values())}")
    counters.add.broadcast(10)  # fire and forget
    print(f"  Counts after:  {list(counters.get_count.call().get().values())}")


if __name__ == "__main__":
    main()
