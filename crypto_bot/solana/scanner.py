from __future__ import annotations

"""Utilities for scanning new Solana tokens using :class:`PoolWatcher`."""

import logging
import os
from typing import Mapping, List

from .watcher import PoolWatcher

logger = logging.getLogger(__name__)


async def get_solana_new_tokens(config: Mapping[str, object]) -> List[str]:
    """Return new Solana token mints using ``config`` options.

    Parameters
    ----------
    config:
        Configuration mapping. ``config["solana_scanner"]`` is used when
        available to retrieve the watcher settings.
    """

    sol_cfg = (
        config.get("solana_scanner") if "solana_scanner" in config else config
    )
    if not isinstance(sol_cfg, Mapping):
        return []

    interval_sec = float(sol_cfg.get("interval_minutes", 0)) * 60.0
    helius_ws_url = os.path.expandvars(str(sol_cfg.get("helius_ws_url", "")))
    websocket_url = helius_ws_url if sol_cfg.get("use_ws", False) else None
    raydium_program_id = sol_cfg.get("raydium_program_id")
    url = sol_cfg.get("url")
    min_liquidity = float(sol_cfg.get("min_liquidity", 0.0))
    watcher = PoolWatcher(
        url,
        interval_sec or None,
        websocket_url,
        raydium_program_id,
        min_liquidity=min_liquidity,
    )

    tokens: List[str] = []
    async for event in watcher.watch():
        tokens.append(event.token_mint)
    watcher.stop()
    return tokens
