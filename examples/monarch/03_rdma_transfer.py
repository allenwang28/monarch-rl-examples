# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchmonarch",
# ]
# ///
"""
03: RDMA Transfer

Demonstrates the control plane vs data plane separation in Monarch.

Run with: uv run examples/03_rdma_transfer.py

Key concepts:
- Control plane: Actor messages are great for coordination (small, flexible)
- Data plane: RDMA is for bulk data transfer (fast, direct memory access)
- RDMABuffer: A handle that represents remote memory - small to pass around,
  but allows pulling large amounts of data efficiently

The pattern:
1. Actor A creates an RDMABuffer from its tensor
2. Actor A sends the handle to Actor B (tiny message via control plane)
3. Actor B calls handle.read_into() to pull the data (bulk transfer via data plane)

Think of RDMABuffer as a "magic pointer" - you pass around a small reference,
and the receiver can use it to efficiently access large remote data.
"""

import time
import torch
from monarch.actor import Actor, endpoint, this_host
from monarch.rdma import RDMABuffer, is_rdma_available


class DataHolder(Actor):
    """An actor that holds data and exposes it via RDMA."""

    def __init__(self, size: int):
        # Create some data (in practice, this would be model weights, gradients, etc.)
        self.data = torch.arange(size, dtype=torch.float32)
        print(f"DataHolder: Created tensor with {size} floats ({self.data.nbytes} bytes)")

    @endpoint
    def get_data_directly(self) -> torch.Tensor:
        """Return the data directly through the control plane.

        This serializes the entire tensor and sends it through actor messages.
        Fine for small data, but inefficient for large tensors.
        """
        return self.data

    @endpoint
    def get_rdma_handle(self) -> RDMABuffer:
        """Return an RDMA handle to the data.

        The handle is tiny (just metadata about where the data lives).
        The receiver can use it to pull the actual data via RDMA.
        """
        # RDMABuffer requires a 1D contiguous byte view
        byte_view = self.data.view(torch.uint8).flatten()
        return RDMABuffer(byte_view)


class DataReceiver(Actor):
    """An actor that receives data from a DataHolder."""

    def __init__(self, size: int):
        # Pre-allocate buffer to receive data into
        self.buffer = torch.zeros(size, dtype=torch.float32)

    @endpoint
    def receive_via_control_plane(self, holder: DataHolder) -> float:
        """Receive data by asking the holder to send it directly.

        The entire tensor travels through the actor message system.
        """
        data = holder.get_data_directly.call_one().get()
        return float(data.sum())

    @endpoint
    def receive_via_rdma(self, holder: DataHolder) -> float:
        """Receive data by getting an RDMA handle and pulling the data.

        Only a tiny handle travels through messages. The bulk data
        is transferred directly via RDMA.
        """
        # Step 1: Get the handle (small message through control plane)
        handle: RDMABuffer = holder.get_rdma_handle.call_one().get()

        # Step 2: Pull the data into our local buffer (bulk transfer via RDMA)
        byte_view = self.buffer.view(torch.uint8).flatten()
        handle.read_into(byte_view).get()

        return float(self.buffer.sum())


def benchmark(name: str, fn, n_iters: int = 10):
    """Run a function n_iters times and report timing."""
    # Warmup
    fn()

    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_iters) * 1000
    print(f"  {name}: {avg_ms:.2f} ms/iter (avg over {n_iters} iters)")
    return avg_ms


def main():
    if not is_rdma_available():
        print("RDMA is not available on this system.")
        print("This example requires RDMA support (typically InfiniBand or RoCE).")
        return

    # Create two processes
    procs = this_host().spawn_procs(per_host={"procs": 2})

    # Spawn actors on different processes
    size = 1_000_000  # 1M floats = 4MB - large enough to see the difference
    holder = procs.slice(procs=0).spawn("holder", DataHolder, size)
    receiver = procs.slice(procs=1).spawn("receiver", DataReceiver, size)

    n_iters = 10
    print(f"\n--- Benchmarking {size:,} floats ({size * 4 / 1024 / 1024:.1f} MB) ---")

    control_plane_ms = benchmark(
        "Control plane (serialize + send)",
        lambda: receiver.receive_via_control_plane.call_one(holder).get(),
        n_iters,
    )

    rdma_ms = benchmark(
        "Data plane (RDMA)",
        lambda: receiver.receive_via_rdma.call_one(holder).get(),
        n_iters,
    )

    print(f"\n--- Results ---")
    if rdma_ms < control_plane_ms:
        speedup = control_plane_ms / rdma_ms
        print(f"  RDMA is {speedup:.1f}x faster")
    else:
        print(f"  Control plane was faster (unusual - may be small data or overhead)")

    print(f"\n--- Network topology note ---")
    print("  This example runs on a single node:")
    print("    - Actor messages travel over Unix sockets (not TCP)")
    print("    - RDMA uses loopback on a single NIC")
    print("  In a multi-host setup:")
    print("    - Actor messages would use TCP")
    print("    - RDMA would use the full network fabric (InfiniBand/RoCE)")
    print("    - The performance gap would likely be larger")

    print(f"\n--- Why RDMA matters ---")
    print("  For large data: RDMA wins because:")
    print("    1. No serialization/deserialization overhead")
    print("    2. Direct memory-to-memory transfer")
    print("    3. Doesn't block the actor's message processing")


if __name__ == "__main__":
    main()
