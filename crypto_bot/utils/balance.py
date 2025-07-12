from __future__ import annotations

"""Utility functions for retrieving wallet balances."""

import asyncio
from typing import Any

from .logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "bot.log")


async def get_usdt_balance(exchange: Any, config: dict | None = None) -> float:
    """Return the free balance for the base currency.

    The helper handles both synchronous and asynchronous ``fetch_balance``
    implementations. ``config`` may specify ``base_currency`` to use instead of
    the default ``USDT``.
    """
    key = (config or {}).get("base_currency", "USDT")
    fetch_fn = getattr(exchange, "fetch_balance", None)
    if fetch_fn is None:
        logger.warning("Exchange has no fetch_balance method")
        return 0.0

    if asyncio.iscoroutinefunction(fetch_fn):
        bal = await fetch_fn()
    else:
        bal = await asyncio.to_thread(fetch_fn)

    value = bal.get(key, {}).get("free", bal.get(key, 0))
    if not value:
        logger.warning("%s balance not found or zero", key)
    return float(value or 0)


async def get_btc_balance(exchange: Any, config: dict | None = None) -> float:
    """Return the free BTC balance using ``get_usdt_balance``."""

    conf = dict(config or {})
    conf.setdefault("base_currency", "BTC")
    return await get_usdt_balance(exchange, conf)
