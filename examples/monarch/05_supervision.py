# Supervision
#
# The __supervise__ hook: How parent actors handle child failures.
#
# When you spawn actors, you become their "owner". If they fail (crash, raise
# an exception), your __supervise__ method is called. You can:
#   - Log the failure and recover
#   - Return True to "handle" it (stops propagation)
#   - Return False/None to let it propagate up the hierarchy
#
# This is how fault tolerance works in Monarch: failures bubble up until
# someone handles them.

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchmonarch",
# ]
# ///

import time
from monarch.actor import Actor, endpoint, current_rank, this_host


class UnreliableWorker(Actor):
    """A worker that might fail."""

    def __init__(self, fail_probability: float = 0.0):
        # current_rank() returns a Point - use .rank to get integer
        self.rank = current_rank().rank
        self.fail_probability = fail_probability
        self.tasks_completed = 0

    @endpoint
    def do_work(self, task_id: int) -> dict:
        """Do some work. Might fail based on fail_probability."""
        import random

        if random.random() < self.fail_probability:
            raise RuntimeError(f"Worker {self.rank} crashed on task {task_id}!")

        self.tasks_completed += 1
        return {"rank": self.rank, "task_id": task_id, "completed": self.tasks_completed}

    @endpoint
    def crash(self) -> None:
        """Deliberately crash this worker."""
        raise RuntimeError(f"Worker {self.rank} deliberately crashed!")


class Supervisor(Actor):
    """
    A supervisor that owns worker actors and handles their failures.
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.failures: list[str] = []
        self.workers = None

        # Spawn workers - we become their owner
        worker_procs = this_host().spawn_procs(per_host={"procs": num_workers})
        self.workers = worker_procs.spawn("workers", UnreliableWorker, fail_probability=0.0)

        print(f"Supervisor: spawned {num_workers} workers")

    def __supervise__(self, failure) -> bool:
        """
        Called when an owned actor fails.

        Args:
            failure: MeshFailure object with details about what failed.
                     Call failure.report() for a readable error message.

        Returns:
            True: Failure is handled, don't propagate further
            False/None: Failure is unhandled, propagate to parent
        """
        # Get a readable error report
        report = failure.report()
        self.failures.append(report)

        print(f"\n[SUPERVISOR] Caught failure!")
        print(f"[SUPERVISOR] Report: {report}")
        print(f"[SUPERVISOR] Total failures so far: {len(self.failures)}")

        # Return True to indicate we've handled it
        # If we returned False/None, it would propagate up
        return True

    @endpoint
    def run_tasks(self, num_tasks: int) -> dict:
        """Run tasks across workers."""
        results = []
        for task_id in range(num_tasks):
            try:
                task_results = self.workers.do_work.call(task_id).get()
                for point, result in task_results.items():
                    results.append(result)
            except Exception as e:
                print(f"[SUPERVISOR] Task {task_id} failed: {e}")

        return {"completed": len(results), "failures": len(self.failures)}

    @endpoint
    def trigger_worker_crash(self) -> str:
        """Tell one worker to crash deliberately."""
        try:
            # Call crash on just the first worker
            self.workers.slice(procs=0).crash.call_one().get()
        except Exception as e:
            return f"Crash triggered, caught: {e}"
        return "Crash triggered"

    @endpoint
    def get_failure_log(self) -> list[str]:
        """Get all recorded failures."""
        return self.failures


def main():
    print("=== Supervision Demo ===\n")

    # Spawn a supervisor (which will spawn workers)
    supervisor_proc = this_host().spawn_procs(per_host={"procs": 1})
    supervisor = supervisor_proc.spawn("supervisor", Supervisor, num_workers=3)

    print("\n--- Running normal tasks ---")
    result = supervisor.run_tasks.call_one(5).get()
    print(f"Completed: {result['completed']} tasks, {result['failures']} failures")

    print("\n--- Triggering a worker crash ---")
    crash_result = supervisor.trigger_worker_crash.call_one().get()
    print(f"Result: {crash_result}")

    # Give supervision a moment to process
    time.sleep(0.5)

    print("\n--- Checking failure log ---")
    failures = supervisor.get_failure_log.call_one().get()
    print(f"Total recorded failures: {len(failures)}")
    for i, failure in enumerate(failures):
        print(f"  Failure {i + 1}: {failure[:100]}...")  # Truncate for readability

    print("\n--- Key Points ---")
    print("1. __supervise__(self, failure) is called when owned actors fail")
    print("2. failure.report() gives a readable error message")
    print("3. Return True to handle the failure, False/None to propagate")
    print("4. This enables building fault-tolerant actor hierarchies")


if __name__ == "__main__":
    main()
