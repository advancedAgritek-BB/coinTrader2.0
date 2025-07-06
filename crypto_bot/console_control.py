from __future__ import annotations

"""Console utilities for starting and stopping the trading bot."""

import asyncio


async def control_loop(state: dict) -> None:
    """Listen for commands and update ``state`` accordingly."""
    print("Commands: start | stop | quit")
    while True:
        cmd = (await asyncio.to_thread(input, "> ")).strip().lower()
        if cmd == "start":
            state["running"] = True
            print("Trading started")
        elif cmd == "stop":
            state["running"] = False
            print("Trading stopped")
        elif cmd in {"quit", "exit"}:
            state["running"] = False
            break
