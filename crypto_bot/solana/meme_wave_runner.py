from __future__ import annotations

import asyncio
from typing import Mapping, Optional

from .watcher import PoolWatcher
from .safety import is_safe
from .score import score_event
from .risk import RiskTracker
from .executor import snipe


async def _run(cfg: Mapping[str, object]) -> None:
    """Background task that watches pools and triggers snipes."""
    pool_cfg = cfg.get("pool", {})
    watcher = PoolWatcher(
        pool_cfg.get("url", ""),
        pool_cfg.get("interval", 5),
        pool_cfg.get("websocket_url"),
        pool_cfg.get("raydium_program_id"),
    )
    tracker = RiskTracker(cfg.get("risk_file", "crypto_bot/logs/sniper_risk.json"))
    safety_cfg = cfg.get("safety", {})
    scoring_cfg = cfg.get("scoring", {})
    risk_cfg = cfg.get("risk", {})
    exec_cfg = cfg.get("execution", {})
    async for event in watcher.watch():
        if not is_safe(event, safety_cfg):
            continue
        score = score_event(event, scoring_cfg)
        if not tracker.allow_snipe(event.token_mint, risk_cfg):
            continue
        tracker.add_snipe(event.token_mint, event.liquidity)
        await snipe(event, score, exec_cfg)


def start_runner(cfg: Mapping[str, object]) -> Optional[asyncio.Task]:
    """Return a task running the meme-wave sniping loop when enabled."""
    if not cfg.get("enabled"):
        return None
    return asyncio.create_task(_run(cfg))
