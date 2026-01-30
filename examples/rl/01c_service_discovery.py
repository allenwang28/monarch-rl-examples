# Service Discovery
#
# Using get_or_spawn_controller for service discovery.
#
# The pattern:
# 1. ServiceRegistry is a singleton controller (found by name)
# 2. Services register themselves on startup
# 3. Any actor can discover services without passing references
#
# This decouples service creation from service usage. In RL, this means
# the trainer can find the generator service, the buffer can find the trainer,
# etc. - all without explicit reference passing.

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchmonarch",
# ]
# ///

from monarch.actor import Actor, endpoint, current_rank, this_host, get_or_spawn_controller


# =============================================================================
# ServiceRegistry: Singleton controller for discovery
# =============================================================================


class ServiceRegistry(Actor):
    """
    Singleton registry for service discovery.

    Found via get_or_spawn_controller("services", ServiceRegistry).
    First caller spawns it, subsequent callers get the existing one.
    """

    def __init__(self):
        self.services: dict[str, Actor] = {}
        print("[REGISTRY] ServiceRegistry spawned")

    @endpoint
    def register(self, name: str, service) -> None:
        """Register a service by name."""
        self.services[name] = service
        print(f"[REGISTRY] Registered '{name}'")

    @endpoint
    def get(self, name: str):
        """Get a service by name."""
        if name not in self.services:
            raise KeyError(f"Service '{name}' not found. Available: {list(self.services.keys())}")
        return self.services[name]

    @endpoint
    def list_services(self) -> list[str]:
        """List all registered service names."""
        return list(self.services.keys())

    @endpoint
    def unregister(self, name: str) -> bool:
        """Unregister a service."""
        if name in self.services:
            del self.services[name]
            print(f"[REGISTRY] Unregistered '{name}'")
            return True
        return False


# =============================================================================
# Example services that register themselves
# =============================================================================


class GeneratorService(Actor):
    """A generator service that registers itself on startup."""

    def __init__(self, name: str = "generators"):
        self.name = name
        self.rank = current_rank().rank
        self.calls = 0

        # Find registry and register ourselves
        registry = get_or_spawn_controller("services", ServiceRegistry).get()
        registry.register.call_one(name, self).get()

    @endpoint
    def generate(self, prompt: str) -> dict:
        """Generate a response."""
        self.calls += 1
        return {
            "service": self.name,
            "rank": self.rank,
            "calls": self.calls,
            "result": f"Generated from '{prompt}'",
        }


class BufferService(Actor):
    """A replay buffer service that registers itself."""

    def __init__(self, name: str = "buffer"):
        self.name = name
        self.buffer: list[dict] = []

        # Register ourselves
        registry = get_or_spawn_controller("services", ServiceRegistry).get()
        registry.register.call_one(name, self).get()

    @endpoint
    def add(self, item: dict) -> int:
        """Add an item to the buffer."""
        self.buffer.append(item)
        return len(self.buffer)

    @endpoint
    def sample(self) -> dict:
        """Sample from the buffer."""
        if not self.buffer:
            return {"error": "Buffer empty"}
        import random
        return random.choice(self.buffer)

    @endpoint
    def size(self) -> int:
        return len(self.buffer)


# =============================================================================
# Trainer: Discovers services by name
# =============================================================================


class Trainer(Actor):
    """
    Trainer that discovers other services by name.
    No references passed in - it finds them itself.
    """

    def __init__(self):
        self.rank = current_rank().rank
        self.step = 0

    @endpoint
    def train_step(self) -> dict:
        """
        One training step:
        1. Find generator service, get a generation
        2. Find buffer service, add to buffer
        3. Sample from buffer for "training"
        """
        self.step += 1

        # Find services by name - no references passed to Trainer!
        registry = get_or_spawn_controller("services", ServiceRegistry).get()

        # Get a generation
        gen_service = registry.get.call_one("generators").get()
        generation = gen_service.generate.call_one(f"prompt_{self.step}").get()

        # Add to buffer
        buffer_service = registry.get.call_one("buffer").get()
        buffer_size = buffer_service.add.call_one(generation).get()

        # Sample for "training"
        sample = buffer_service.sample.call_one().get()

        return {
            "trainer_rank": self.rank,
            "step": self.step,
            "generation": generation["result"],
            "buffer_size": buffer_size,
            "sampled": sample["result"] if "result" in sample else sample,
        }


# =============================================================================
# Demo
# =============================================================================


def main():
    print("=== Service Discovery Demo ===\n")

    # Spawn services - they register themselves
    print("--- Spawning services ---")
    gen_proc = this_host().spawn_procs(per_host={"procs": 1})
    gen_service = gen_proc.spawn("gen_service", GeneratorService, "generators")

    buffer_proc = this_host().spawn_procs(per_host={"procs": 1})
    buffer_service = buffer_proc.spawn("buffer_service", BufferService, "buffer")

    # Check what's registered
    registry = get_or_spawn_controller("services", ServiceRegistry).get()
    services = registry.list_services.call_one().get()
    print(f"\nRegistered services: {services}")

    # Spawn trainer - it discovers services by name
    print("\n--- Spawning trainer ---")
    trainer_proc = this_host().spawn_procs(per_host={"procs": 1})
    trainer = trainer_proc.spawn("trainer", Trainer)

    # Run some training steps
    print("\n--- Training steps ---")
    for i in range(5):
        result = trainer.train_step.call_one().get()
        print(f"Step {result['step']}: generated='{result['generation']}', "
              f"buffer_size={result['buffer_size']}")

    # Show buffer contents
    print(f"\n--- Buffer has {buffer_service.size.call_one().get()} items ---")

    print("\n--- Key Points ---")
    print("1. ServiceRegistry is a singleton via get_or_spawn_controller")
    print("2. Services register themselves on startup")
    print("3. Trainer finds services by name - no reference passing")
    print("4. This decouples service creation from usage")
    print("\nIn real RL: trainer ↔ generators ↔ buffer all find each other by name")


if __name__ == "__main__":
    main()
