from __future__ import annotations

"""Asynchronous runner for Solana meme-wave sniping."""

import asyncio
from typing import Mapping

from .watcher import PoolWatcher, NewPoolEvent
from .score import score_event
from . import executor


async def run(config: Mapping[str, object]) -> None:
    """Run the meme-wave sniper loop using ``config`` options."""
    pool_cfg = config.get("pool", {}) if isinstance(config, Mapping) else {}
    url = str(pool_cfg.get("url", ""))
    interval = float(pool_cfg.get("interval", 5))
    ws_url = pool_cfg.get("websocket_url")
    program_id = pool_cfg.get("raydium_program_id")
    watcher = PoolWatcher(url, interval, ws_url, program_id)

    try:
        async for event in watcher.watch():
            score = score_event(event, config.get("scoring", {}))
            await executor.snipe(event, score, config.get("execution", {}))
    except asyncio.CancelledError:
        watcher.stop()
        raise
