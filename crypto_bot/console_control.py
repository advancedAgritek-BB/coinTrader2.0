from __future__ import annotations

"""Console utilities for starting and stopping the trading bot."""

import asyncio
from typing import Dict, Any

from crypto_bot.strategy import bounce_scalper


async def control_loop(state: Dict[str, Any]) -> None:
    """Listen for commands and update ``state`` accordingly."""
    print("Commands: start | stop | quit | b (bounce scalper)")
    try:
        while True:
            cmd = (await asyncio.to_thread(input, "> ")).strip().lower()
            if cmd == "start":
                state["running"] = True
                print("Trading started")
            elif cmd == "stop":
                state["running"] = False
                print("Trading stopped")
            elif cmd == "b":
                bounce_scalper.trigger_once()
                print("Bounce scalper triggered")
            elif cmd in {"quit", "exit"}:
                state["running"] = False
                break
    except asyncio.CancelledError:
        state["running"] = False
        raise

