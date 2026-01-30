# Service with Mesh Replicas
#
# Extends 01_simple_service.py to support mesh replicas.
#
# In real RL systems, each "replica" isn't a single actor - it's an ActorMesh.
# For example, a generator might use 2 GPUs for tensor parallelism. A service
# with 4 generator replicas would use 8 GPUs total (4 replicas × 2 GPUs each).
#
# This pattern generalizes to multi-host: the replica_procs mesh could span
# multiple hosts, and each replica would be a slice of that mesh.
#
# Key difference from 01_simple_service.py:
# - Service receives a ProcMesh of resources
# - Slices it into replica-sized chunks
# - Each replica is an ActorMesh, not a single actor

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchmonarch",
# ]
# ///

from monarch.actor import Actor, endpoint, current_rank, this_host


# =============================================================================
# Worker: Same as before, but now part of a mesh
# =============================================================================


class Worker(Actor):
    """
    A worker that's part of a replica mesh.
    In real RL, this might do tensor-parallel inference.
    """

    def __init__(self):
        self.rank = current_rank().rank
        self.calls = 0

    @endpoint
    def process(self, data: str) -> dict:
        """Process data. In a real mesh, workers would coordinate."""
        self.calls += 1
        return {
            "worker_rank": self.rank,
            "calls": self.calls,
            "result": f"Processed '{data}' (worker {self.rank})",
        }

    @endpoint
    def get_rank(self) -> int:
        return self.rank


# =============================================================================
# MeshService: Owns mesh replicas, routes requests
# =============================================================================


class MeshService(Actor):
    """
    Service where each replica is an ActorMesh.

    The service:
    1. Receives a ProcMesh of resources
    2. Slices it into replica-sized chunks
    3. Spawns an ActorMesh on each chunk
    4. Routes requests to healthy replica meshes
    """

    def __init__(
        self,
        worker_class: type,
        replica_procs,  # ProcMesh to slice into replicas
        procs_per_replica: int,
    ):
        self.worker_class = worker_class
        self.procs_per_replica = procs_per_replica

        # Calculate how many replicas we can make
        # Use len() to get total procs, or .size('procs') for specific dimension
        total_procs = len(replica_procs)
        self.num_replicas = total_procs // procs_per_replica

        if self.num_replicas == 0:
            raise ValueError(
                f"Not enough procs: have {total_procs}, need {procs_per_replica} per replica"
            )

        # Slice the proc mesh and spawn actor meshes
        self.replicas: list = []  # List of ActorMesh objects
        for i in range(self.num_replicas):
            start = i * procs_per_replica
            end = start + procs_per_replica

            # Slice out this replica's procs
            replica_slice = replica_procs.slice(procs=slice(start, end))

            # Spawn actors on this slice
            replica_mesh = replica_slice.spawn(f"replica_{i}", worker_class)
            self.replicas.append(replica_mesh)

        # Health tracking
        self.healthy = set(range(self.num_replicas))
        self.next_idx = 0

        print(f"MeshService: {self.num_replicas} replicas × {procs_per_replica} procs each")

    def __supervise__(self, failure) -> bool:
        """Called when a replica fails."""
        report = failure.report()
        print(f"[MESH SERVICE] Failure: {report[:100]}...")
        return True

    @endpoint
    def get_replica(self):
        """
        Get a healthy replica mesh (round-robin).
        Returns an ActorMesh - caller decides how to call it.
        """
        if not self.healthy:
            raise RuntimeError("No healthy replicas!")

        healthy_list = sorted(self.healthy)
        idx = self.next_idx % len(healthy_list)
        self.next_idx += 1

        replica_idx = healthy_list[idx]
        return self.replicas[replica_idx]

    @endpoint
    def get_replica_by_index(self, idx: int):
        """Get a specific replica by index."""
        if idx not in self.healthy:
            raise RuntimeError(f"Replica {idx} is not healthy")
        return self.replicas[idx]

    @endpoint
    def mark_unhealthy(self, replica_idx: int) -> None:
        """Mark a replica as unhealthy."""
        if replica_idx in self.healthy:
            self.healthy.remove(replica_idx)
            print(f"[MESH SERVICE] Replica {replica_idx} marked unhealthy. "
                  f"Healthy: {len(self.healthy)}/{self.num_replicas}")

    @endpoint
    def mark_healthy(self, replica_idx: int) -> None:
        """Mark a replica as healthy."""
        if replica_idx < self.num_replicas:
            self.healthy.add(replica_idx)
            print(f"[MESH SERVICE] Replica {replica_idx} marked healthy.")

    @endpoint
    def get_health_status(self) -> dict:
        return {
            "total_replicas": self.num_replicas,
            "procs_per_replica": self.procs_per_replica,
            "healthy": len(self.healthy),
            "healthy_indices": sorted(self.healthy),
        }


# =============================================================================
# Demo
# =============================================================================


def main():
    print("=== Mesh Service Demo ===\n")

    # Reserve resources: 8 procs total
    # We'll slice into 4 replicas of 2 procs each
    print("--- Reserving resources ---")
    replica_procs = this_host().spawn_procs(per_host={"procs": 8})
    print(f"Reserved 8 procs")

    # Spawn service coordinator (lightweight)
    service_proc = this_host().spawn_procs(per_host={"procs": 1})

    print("\n--- Spawning MeshService ---")
    service = service_proc.spawn("mesh_service", MeshService,
        worker_class=Worker,
        replica_procs=replica_procs,
        procs_per_replica=2,
    )

    status = service.get_health_status.call_one().get()
    print(f"Created {status['total_replicas']} replicas, "
          f"{status['procs_per_replica']} procs each")

    print("\n--- Using the service ---")
    for i in range(4):
        # Get a replica mesh
        replica = service.get_replica.call_one().get()

        # Call on the mesh - we get results from all workers in the replica
        # Using .call() returns a ValueMesh with results from each worker
        results = replica.process.call(f"request_{i}").get()

        print(f"Request {i}:")
        for point, result in results.items():
            print(f"  Worker {result['worker_rank']}: {result['result']}")

    print("\n--- Health tracking ---")
    # Mark replica 0 as unhealthy
    service.mark_unhealthy.call_one(0).get()

    status = service.get_health_status.call_one().get()
    print(f"Healthy: {status['healthy']}/{status['total_replicas']} "
          f"- indices: {status['healthy_indices']}")

    # Subsequent calls will skip replica 0
    print("\nNext 3 calls (should skip replica 0):")
    for i in range(3):
        replica = service.get_replica.call_one().get()
        results = replica.get_rank.call().get()
        ranks = [r for _, r in results.items()]
        print(f"  Got replica with worker ranks: {ranks}")

    print("\n--- Key Points ---")
    print("1. Service receives a ProcMesh, slices into replica chunks")
    print("2. Each replica is an ActorMesh (multiple workers)")
    print("3. get_replica() returns an ActorMesh - caller uses .call()/.broadcast()")
    print("4. Same health tracking pattern as single-actor service")
    print("\nThis pattern generalizes to multi-host:")
    print("  replica_procs = host_mesh.spawn_procs(per_host={'gpus': 8})")
    print("  # Slicing across hosts gives you distributed replicas")


if __name__ == "__main__":
    main()
