"""Execution helpers for Solana sniping."""

from __future__ import annotations

import asyncio
from typing import Mapping, Dict

from .watcher import NewPoolEvent


async def snipe(event: NewPoolEvent, score: float, cfg: Mapping[str, object]) -> Dict:
    """Execute a snipe trade for ``event`` using :func:`solana_trading.sniper_trade`."""

    from solana_trading import sniper_trade

    details = {
        "pool": event.pool_address,
        "mint": event.token_mint,
        "creator": event.creator,
        "liquidity": event.liquidity,
        "tx_count": event.tx_count,
        "freeze_authority": event.freeze_authority,
        "mint_authority": event.mint_authority,
        "timestamp": event.timestamp,
        "score": score,
    }

    return await sniper_trade(details, cfg)
