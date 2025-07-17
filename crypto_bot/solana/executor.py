"""Execution helpers for Solana sniping."""

from __future__ import annotations

import asyncio
from typing import Mapping, Dict

from .watcher import NewPoolEvent


async def snipe(event: NewPoolEvent, score: float, cfg: Mapping[str, object]) -> Dict:
    """Execute a snipe trade for ``event`` using :func:`crypto_bot.solana_trading.sniper_trade`."""

    from crypto_bot.solana_trading import sniper_trade

    wallet = str(cfg.get("wallet_address", ""))
    base_token = str(cfg.get("base_token", "USDC"))
    amount = float(cfg.get("amount", 0))

    return await sniper_trade(
        wallet,
        base_token,
        event.token_mint,
        amount,
        dry_run=bool(cfg.get("dry_run", True)),
        slippage_bps=int(cfg.get("slippage_bps", 50)),
        notifier=cfg.get("notifier"),
    )
