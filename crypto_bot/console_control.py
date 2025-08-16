from __future__ import annotations

"""Console utilities for starting and stopping the trading bot."""

import asyncio
import logging
from typing import Dict, Any


async def control_loop(state: Dict[str, Any]) -> None:
    """Listen for commands and update ``state`` accordingly."""
    print("Commands: start | stop | reload | panic sell | quit")
    try:
        while True:
            try:
                cmd = (await asyncio.to_thread(input, "> ")).strip().lower()
            except EOFError:
                logging.warning("EOF on console input; exiting control loop")
                break
            if cmd == "start":
                state["running"] = True
                print("Trading started")
            elif cmd == "stop":
                state["running"] = False
                print("Trading stopped")
            elif cmd == "reload":
                state["reload"] = True
                print("Reloading config")
            elif cmd in {"panic", "panic sell", "panic_sell"}:
                state["liquidate_all"] = True
                print("Liquidation scheduled")
            elif cmd in {"quit", "exit"}:
                state["running"] = False
                break
    except asyncio.CancelledError:
        state["running"] = False
        raise

