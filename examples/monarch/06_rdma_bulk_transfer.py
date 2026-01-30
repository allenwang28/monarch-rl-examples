# RDMA Bulk Transfer
#
# Memory registration (MR) is expensive. This example shows how to amortize
# that cost across multiple transfers by keeping handles alive.
#
# We compare three approaches over multiple "training steps":
#
# 1. Naive: Create new RDMABuffer every transfer (pay MR cost each time)
# 2. Contiguous: One buffer, one MR, reuse across all steps
# 3. Scattered + RDMAAction: Multiple buffers, register once, batch transfers
#
# Watch how the naive approach is slow every time, while smart approaches
# pay the registration cost once and then fly.
#
# NOTE: As of monarch 0.2.2, RDMAAction is a Python-level stub. The batching
# logic will be lowered to the Rust layer in a future release for better
# performance. For now, it's a convenience API for organizing operations.

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchmonarch",
# ]
# ///

import time
import torch
from monarch.actor import Actor, endpoint, current_rank, this_host
from monarch.rdma import RDMABuffer, RDMAAction


# =============================================================================
# Approach 1: Naive (re-register every transfer)
# =============================================================================


class NaiveSender(Actor):
    """Creates a new RDMABuffer for every transfer. Expensive!"""

    def __init__(self, layer_sizes: list[int]):
        self.layer_sizes = layer_sizes
        self.layers = [torch.zeros(size, dtype=torch.float32) for size in layer_sizes]
        # Initialize with recognizable values
        for i, layer in enumerate(self.layers):
            layer.fill_(float(i + 1))
        print(f"NaiveSender: {len(layer_sizes)} layers (will register fresh each time)")

    @endpoint
    def get_fresh_handles(self) -> list[tuple[int, RDMABuffer]]:
        """Create NEW RDMABuffer handles every time. This is expensive!"""
        handles = []
        for size, layer in zip(self.layer_sizes, self.layers):
            # New registration every call!
            byte_view = layer.view(torch.uint8).flatten()
            handle = RDMABuffer(byte_view)
            handles.append((size, handle))
        return handles


class NaiveReceiver(Actor):
    """Receives from naive sender - pays MR cost every step."""

    def __init__(self, layer_sizes: list[int]):
        self.layer_sizes = layer_sizes
        self.layers = [torch.zeros(size, dtype=torch.float32) for size in layer_sizes]
        self.rank = current_rank().rank

    @endpoint
    def receive_step(self, sender: NaiveSender) -> dict:
        start = time.perf_counter()

        # Get fresh handles (sender registers new MRs)
        handles = sender.get_fresh_handles.call_one().get()

        # Transfer each layer
        for i, (size, handle) in enumerate(handles):
            byte_view = self.layers[i].view(torch.uint8).flatten()
            handle.read_into(byte_view).get()

        elapsed_ms = (time.perf_counter() - start) * 1000
        checksum = sum(float(layer.sum()) for layer in self.layers)
        return {"rank": self.rank, "elapsed_ms": elapsed_ms, "checksum": checksum}


# =============================================================================
# Approach 2: Smart Contiguous (register once, reuse)
# =============================================================================


class ContiguousSender(Actor):
    """One buffer, one MR, registered at startup."""

    def __init__(self, layer_sizes: list[int]):
        self.layer_sizes = layer_sizes
        total_size = sum(layer_sizes)

        # One contiguous buffer
        self.buffer = torch.zeros(total_size, dtype=torch.float32)
        offset = 0
        for i, size in enumerate(layer_sizes):
            self.buffer[offset : offset + size].fill_(float(i + 1))
            offset += size

        # Register ONCE at startup
        byte_view = self.buffer.view(torch.uint8).flatten()
        self.handle = RDMABuffer(byte_view)
        print(f"ContiguousSender: {total_size} floats, 1 MR (registered once)")

    @endpoint
    def get_handle(self) -> tuple[int, RDMABuffer]:
        """Return the same handle every time. No new registration!"""
        return (len(self.buffer), self.handle)


class ContiguousReceiver(Actor):
    """Receives from contiguous sender - fast after first step."""

    def __init__(self, total_size: int):
        self.buffer = torch.zeros(total_size, dtype=torch.float32)
        self.rank = current_rank().rank

    @endpoint
    def receive_step(self, sender: ContiguousSender) -> dict:
        start = time.perf_counter()

        size, handle = sender.get_handle.call_one().get()
        byte_view = self.buffer.view(torch.uint8).flatten()
        handle.read_into(byte_view).get()

        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"rank": self.rank, "elapsed_ms": elapsed_ms, "checksum": float(self.buffer.sum())}


# =============================================================================
# Approach 3: Smart Scattered + RDMAAction (register once, batch transfers)
# =============================================================================


class ScatteredSender(Actor):
    """Multiple buffers, each registered once at startup."""

    def __init__(self, layer_sizes: list[int]):
        self.layer_sizes = layer_sizes
        self.layers = []
        self.handles = []

        for i, size in enumerate(layer_sizes):
            layer = torch.zeros(size, dtype=torch.float32)
            layer.fill_(float(i + 1))
            self.layers.append(layer)

            # Register ONCE at startup
            byte_view = layer.view(torch.uint8).flatten()
            self.handles.append(RDMABuffer(byte_view))

        print(f"ScatteredSender: {len(layer_sizes)} layers, {len(self.handles)} MRs (registered once)")

    @endpoint
    def get_handles(self) -> list[tuple[int, RDMABuffer]]:
        """Return the same handles every time. No new registration!"""
        return [(size, handle) for size, handle in zip(self.layer_sizes, self.handles)]


class ScatteredReceiver(Actor):
    """Receives from scattered sender with RDMAAction batching."""

    def __init__(self, layer_sizes: list[int]):
        self.layer_sizes = layer_sizes
        self.layers = [torch.zeros(size, dtype=torch.float32) for size in layer_sizes]
        self.rank = current_rank().rank

    @endpoint
    def receive_step(self, sender: ScatteredSender) -> dict:
        start = time.perf_counter()

        handles = sender.get_handles.call_one().get()

        # Batch all transfers with RDMAAction
        action = RDMAAction()
        for i, (size, handle) in enumerate(handles):
            byte_view = self.layers[i].view(torch.uint8).flatten()
            action.read_into(handle, byte_view)

        action.submit().get()

        elapsed_ms = (time.perf_counter() - start) * 1000
        checksum = sum(float(layer.sum()) for layer in self.layers)
        return {"rank": self.rank, "elapsed_ms": elapsed_ms, "checksum": checksum}


# =============================================================================
# Demo: Compare across multiple steps
# =============================================================================


def run_steps(name: str, receiver, sender, num_steps: int = 5):
    """Run multiple transfer steps and report timing."""
    print(f"\n{name}:")
    times = []

    for step in range(num_steps):
        results = receiver.receive_step.call_one(sender).get()
        times.append(results["elapsed_ms"])
        print(f"  Step {step + 1}: {results['elapsed_ms']:.2f}ms")

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.2f}ms")
    return times


def main():
    print("=== RDMA Registration Amortization Demo ===")
    print("Comparing 3 approaches over multiple 'training steps'\n")

    layer_sizes = [1000, 5000, 2000]  # 8000 floats total
    total_size = sum(layer_sizes)
    num_steps = 5

    # Setup all senders
    sender_procs = this_host().spawn_procs(per_host={"procs": 1})

    print("--- Setup ---")
    naive_sender = sender_procs.spawn("naive_sender", NaiveSender, layer_sizes)
    cont_sender = sender_procs.spawn("cont_sender", ContiguousSender, layer_sizes)
    scat_sender = sender_procs.spawn("scat_sender", ScatteredSender, layer_sizes)

    # Setup all receivers
    receiver_procs = this_host().spawn_procs(per_host={"procs": 1})
    naive_receiver = receiver_procs.spawn("naive_recv", NaiveReceiver, layer_sizes)
    cont_receiver = receiver_procs.spawn("cont_recv", ContiguousReceiver, total_size)
    scat_receiver = receiver_procs.spawn("scat_recv", ScatteredReceiver, layer_sizes)

    print(f"\n--- Running {num_steps} steps each ---")

    naive_times = run_steps("Naive (re-register each step)", naive_receiver, naive_sender, num_steps)
    cont_times = run_steps("Contiguous (register once)", cont_receiver, cont_sender, num_steps)
    scat_times = run_steps("Scattered + RDMAAction (register once)", scat_receiver, scat_sender, num_steps)

    print("\n=== Summary ===")
    print("Naive: Pays MR registration cost EVERY step")
    print("Smart: Pays MR cost once at startup, subsequent steps are fast")
    print()
    print("Key insight: Keep your RDMABuffer handles alive across training steps!")
    print("This is why we register in __init__, not in the transfer endpoint.")


if __name__ == "__main__":
    main()
