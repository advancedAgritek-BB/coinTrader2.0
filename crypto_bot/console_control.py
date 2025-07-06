import asyncio

async def control_loop(state: dict) -> None:
    """Simple console control loop to pause or stop the bot."""
    while True:
        try:
            cmd = await asyncio.to_thread(input, "")
        except EOFError:
            break
        if cmd is None:
            continue
        cmd = cmd.strip().lower()
        if cmd in {"stop", "pause"}:
            state["running"] = False
            print("Bot paused.")
        elif cmd in {"start", "resume"}:
            state["running"] = True
            print("Bot resumed.")
from __future__ import annotations

"""Console control loop for starting and stopping the trading bot."""

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
