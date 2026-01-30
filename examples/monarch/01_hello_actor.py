# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchmonarch",
# ]
# ///
"""
01: Hello Actor

The most basic Monarch example. We spawn an actor, call an endpoint, and get a result.

Run with: uv run examples/01_hello_actor.py

Key concepts:
- Actor: An isolated unit with state and methods (endpoints)
- endpoint: A method you can call remotely
- spawn: Create an actor instance
- .get(): Block and wait for the result (sync API)
"""

import torch
from monarch.actor import Actor, endpoint, this_proc


class Greeter(Actor):
    """A simple actor that says hello and knows which device it's on."""

    @endpoint
    def greet(self, name: str) -> str:
        if torch.cuda.is_available():
            device = f"GPU {torch.cuda.current_device()}"
        else:
            device = "CPU"
        return f"Hello {name}! I'm running on {device}."


def main():
    # Spawn the actor on this process
    greeter = this_proc().spawn("greeter", Greeter)

    # Call the endpoint and get the result
    # .get() blocks until the result is ready
    result = greeter.greet.call_one("world").get()

    print(result)


if __name__ == "__main__":
    main()
