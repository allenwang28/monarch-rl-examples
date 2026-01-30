# Simple Service
#
# A Service wraps multiple worker replicas and provides:
# - Round-robin routing to healthy replicas
# - Failure detection via __supervise__
# - Health tracking so callers can mark replicas unhealthy
#
# This example uses single-actor replicas. See 01b_service_mesh_replicas.py
# for the pattern with mesh replicas (which scales to multi-host).

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchmonarch",
# ]
# ///

import random
from monarch.actor import Actor, endpoint, current_rank, this_host


# =============================================================================
# Worker: The actual work happens here
# =============================================================================


class Worker(Actor):
    """A worker that does some computation. Might fail."""

    def __init__(self, fail_rate: float = 0.0):
        self.rank = current_rank().rank
        self.fail_rate = fail_rate
        self.calls = 0

    @endpoint
    def process(self, data: str) -> dict:
        """Process some data. Might fail based on fail_rate."""
        self.calls += 1

        if random.random() < self.fail_rate:
            raise RuntimeError(f"Worker {self.rank} failed on call {self.calls}!")

        return {
            "worker": self.rank,
            "calls": self.calls,
            "result": f"Processed '{data}' by worker {self.rank}",
        }

    @endpoint
    def get_id(self) -> int:
        return self.rank


# =============================================================================
# Service: Owns workers, routes requests, handles failures
# =============================================================================


class Service(Actor):
    """
    Wraps worker replicas with routing and health tracking.

    The Service:
    1. Spawns worker replicas at startup
    2. Tracks which replicas are healthy
    3. Routes requests round-robin to healthy replicas
    4. Detects failures via __supervise__
    """

    def __init__(self, worker_class: type, num_replicas: int, *worker_args, **worker_kwargs):
        self.worker_class = worker_class
        self.num_replicas = num_replicas
        self.worker_args = worker_args
        self.worker_kwargs = worker_kwargs

        # Spawn worker replicas
        worker_procs = this_host().spawn_procs(per_host={"procs": num_replicas})
        self.workers = worker_procs.spawn("workers", worker_class, *worker_args, **worker_kwargs)

        # Track health: set of healthy worker ranks
        self.healthy = set(range(num_replicas))

        # Round-robin index
        self.next_idx = 0

        print(f"Service: spawned {num_replicas} replicas")

    def __supervise__(self, failure) -> bool:
        """Called when a worker fails. Mark it unhealthy."""
        report = failure.report()
        print(f"[SERVICE] Failure detected: {report[:100]}...")

        # In a real system, we'd parse the failure to identify which worker
        # For now, we rely on explicit mark_unhealthy calls
        return True  # Handled

    @endpoint
    def get_replica(self) -> Actor:
        """
        Get a healthy replica (round-robin selection).
        Returns the worker actor reference.
        """
        if not self.healthy:
            raise RuntimeError("No healthy replicas available!")

        # Convert to sorted list for consistent ordering
        healthy_list = sorted(self.healthy)

        # Round-robin selection
        idx = self.next_idx % len(healthy_list)
        self.next_idx += 1

        replica_rank = healthy_list[idx]

        # Return the specific worker using slice
        return self.workers.slice(procs=replica_rank)

    @endpoint
    def mark_unhealthy(self, replica_rank: int) -> None:
        """Mark a replica as unhealthy. Called by user after failure."""
        if replica_rank in self.healthy:
            self.healthy.remove(replica_rank)
            print(f"[SERVICE] Marked replica {replica_rank} as unhealthy. "
                  f"Healthy: {len(self.healthy)}/{self.num_replicas}")

    @endpoint
    def mark_healthy(self, replica_rank: int) -> None:
        """Mark a replica as healthy again (after recovery)."""
        if replica_rank < self.num_replicas:
            self.healthy.add(replica_rank)
            print(f"[SERVICE] Marked replica {replica_rank} as healthy. "
                  f"Healthy: {len(self.healthy)}/{self.num_replicas}")

    @endpoint
    def get_health_status(self) -> dict:
        """Get current health status."""
        return {
            "total": self.num_replicas,
            "healthy": len(self.healthy),
            "healthy_ranks": sorted(self.healthy),
        }


# =============================================================================
# Demo: Using the service with retry logic
# =============================================================================


def call_with_retry(service, method: str, *args, max_retries: int = 3, **kwargs):
    """
    Call a method on a service replica with retry logic.

    This is the pattern for using a Service:
    1. Get a replica from the service
    2. Try the call
    3. On failure, mark unhealthy and retry with a different replica
    """
    last_error = None

    for attempt in range(max_retries):
        # Get a healthy replica
        try:
            replica = service.get_replica.call_one().get()
        except RuntimeError as e:
            print(f"[CALLER] No healthy replicas: {e}")
            raise

        # Get the replica's rank for health tracking
        replica_rank = replica.get_id.call_one().get()

        try:
            # Call the method
            method_fn = getattr(replica, method)
            result = method_fn.call_one(*args, **kwargs).get()
            return result

        except Exception as e:
            print(f"[CALLER] Attempt {attempt + 1} failed on replica {replica_rank}: {e}")
            last_error = e

            # Mark this replica as unhealthy
            service.mark_unhealthy.call_one(replica_rank).get()

    raise RuntimeError(f"All {max_retries} attempts failed. Last error: {last_error}")


def main():
    print("=== Simple Service Demo ===\n")

    # Spawn the service (which spawns workers internally)
    service_proc = this_host().spawn_procs(per_host={"procs": 1})
    service = service_proc.spawn("my_service", Service,
        worker_class=Worker,
        num_replicas=4,
        fail_rate=0.3,  # 30% chance of failure for demo
    )

    print("\n--- Initial health status ---")
    status = service.get_health_status.call_one().get()
    print(f"Healthy: {status['healthy']}/{status['total']} - ranks: {status['healthy_ranks']}")

    print("\n--- Making calls with retry logic ---")
    for i in range(8):
        try:
            result = call_with_retry(service, "process", f"request_{i}")
            print(f"Request {i}: {result['result']}")
        except RuntimeError as e:
            print(f"Request {i}: FAILED - {e}")

        # Check health after each request
        status = service.get_health_status.call_one().get()
        if status['healthy'] < status['total']:
            print(f"  (Healthy: {status['healthy']}/{status['total']})")

    print("\n--- Final health status ---")
    status = service.get_health_status.call_one().get()
    print(f"Healthy: {status['healthy']}/{status['total']} - ranks: {status['healthy_ranks']}")

    print("\n--- Key Points ---")
    print("1. Service owns workers and tracks health")
    print("2. get_replica() returns a healthy worker (round-robin)")
    print("3. On failure, caller marks replica unhealthy and retries")
    print("4. __supervise__ detects failures (for logging/alerting)")
    print("\nSee 01b_service_mesh_replicas.py for mesh replicas (scales to multi-host)")


if __name__ == "__main__":
    main()
