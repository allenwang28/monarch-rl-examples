# Controllers
#
# get_or_spawn_controller pattern. Singleton actors, service discovery by name.
#
# The problem: You have multiple actors that all need to find the same
# coordinator (metrics aggregator, config store, parameter server).
# Passing references around is tedious. You want service discovery by name.
#
# The solution: get_or_spawn_controller("name", ActorClass) returns the same
# actor instance to all callers. First caller spawns it, subsequent callers
# get the existing one.

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchmonarch",
# ]
# ///

from monarch.actor import Actor, endpoint, current_rank, this_host, get_or_spawn_controller


class MetricsAggregator(Actor):
    """
    A singleton actor that collects metrics from all workers.
    Only one instance exists, found by name from anywhere.
    """

    def __init__(self):
        self.metrics: dict[int, list[float]] = {}
        print("MetricsAggregator spawned!")

    @endpoint
    def report(self, worker_id: int, value: float) -> int:
        """Workers call this to report metrics. Returns total reports received."""
        if worker_id not in self.metrics:
            self.metrics[worker_id] = []
        self.metrics[worker_id].append(value)
        total = sum(len(v) for v in self.metrics.values())
        return total

    @endpoint
    def get_summary(self) -> dict:
        """Get aggregated metrics from all workers."""
        return {
            "num_workers": len(self.metrics),
            "reports_per_worker": {k: len(v) for k, v in self.metrics.items()},
            "averages": {k: sum(v) / len(v) for k, v in self.metrics.items() if v},
        }


class Worker(Actor):
    """
    A worker that finds the MetricsAggregator by name and reports to it.
    No reference passed in - it discovers the aggregator itself.
    """

    def __init__(self):
        # current_rank() returns a Point object with coordinates
        # Use .rank to get the integer rank
        self.rank = current_rank().rank

    @endpoint
    def do_work(self) -> str:
        # Find the metrics aggregator by name - no reference needed!
        # If it doesn't exist yet, this spawns it. If it does, we get the existing one.
        aggregator = get_or_spawn_controller("metrics", MetricsAggregator).get()

        # Simulate doing work and reporting metrics
        for i in range(3):
            value = self.rank * 10.0 + i
            total = aggregator.report.call_one(self.rank, value).get()

        return f"Worker {self.rank} reported 3 values"


def main():
    # Spawn 4 workers
    procs = this_host().spawn_procs(per_host={"procs": 4})
    workers = procs.spawn("workers", Worker)

    print("=== Controller Pattern Demo ===\n")

    # Each worker independently finds the aggregator and reports metrics
    # They all find the SAME aggregator instance
    print("Workers doing work (each finds aggregator by name)...")
    results = workers.do_work.call().get()
    for point, result in results.items():
        print(f"  {result}")

    print()

    # We can also find the aggregator from here
    # Same instance - we'll see all the metrics the workers reported
    aggregator = get_or_spawn_controller("metrics", MetricsAggregator).get()
    summary = aggregator.get_summary.call_one().get()

    print("Aggregator summary (single instance saw all reports):")
    print(f"  Workers reporting: {summary['num_workers']}")
    print(f"  Reports per worker: {summary['reports_per_worker']}")
    print(f"  Averages: {summary['averages']}")

    print()
    print("Key insight: 'MetricsAggregator spawned!' printed only ONCE.")
    print("All workers and the main process found the same instance by name.")


if __name__ == "__main__":
    main()
