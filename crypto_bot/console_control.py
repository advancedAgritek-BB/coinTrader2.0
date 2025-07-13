from __future__ import annotations

"""Console utilities for starting and stopping the trading bot."""

import asyncio
from typing import Dict, Any


async def control_loop(state: Dict[str, Any]) -> None:
    """Listen for commands and update ``state`` accordingly."""
    print("Commands: start | stop | reload | quit")
    try:
        while True:
            cmd = (await asyncio.to_thread(input, "> ")).strip().lower()
            if cmd == "start":
                state["running"] = True
                print("Trading started")
            elif cmd == "stop":
                state["running"] = False
                print("Trading stopped")
            elif cmd == "reload":
                state["reload"] = True
                print("Reloading config")
            elif cmd in {"quit", "exit"}:
                state["running"] = False
                break
    except asyncio.CancelledError:
        state["running"] = False
        raise

