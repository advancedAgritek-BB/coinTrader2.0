"""Position exit helpers."""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Mapping, Dict, Awaitable
from typing import Callable, Mapping, Dict, Optional

from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor


async def monitor_price(
    price_feed: Callable[[], float | Awaitable[float]],
    entry_price: float,
    cfg: Mapping[str, float],
    *,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
    mempool_cfg: Optional[Mapping[str, float]] = None,
) -> Dict:
    """Monitor price until exit conditions are met.

    Parameters
    ----------
    mempool_monitor:
        Optional :class:`~crypto_bot.execution.solana_mempool.SolanaMempoolMonitor`
        used to abort when priority fees spike.
    mempool_cfg:
        Configuration controlling the mempool check.
    """

    take_profit = float(cfg.get("take_profit_pct", 0))
    trailing = float(cfg.get("trailing_stop_pct", 0))
    timeout = float(cfg.get("timeout", 60))
    poll = float(cfg.get("poll_interval", 1))
    profit_tp = float(cfg.get("profit_tp_pct", 10))
    mp_cfg = mempool_cfg or {}
    mempool_thr = float(mp_cfg.get("suspicious_fee_threshold", 0.0))

    peak = entry_price
    start = time.time()
    price = entry_price
    while time.time() - start < timeout:
        await asyncio.sleep(poll)
        if mempool_monitor and mp_cfg.get("enabled") and mempool_monitor.is_suspicious(mempool_thr):
            return {"exit_price": price_feed(), "reason": "mempool_spike"}
        price = price_feed()
        if asyncio.iscoroutine(price):
            price = await price
        if price > peak:
            peak = price
        change_pct = (price - entry_price) / entry_price * 100
        if change_pct >= profit_tp:
            return {"exit_price": price, "reason": "tp"}
        if take_profit and price >= entry_price * (1 + take_profit / 100):
            return {"exit_price": price, "reason": "tp"}
        if trailing and price <= peak * (1 - trailing / 100):
            return {"exit_price": price, "reason": "trailing"}
    return {"exit_price": price, "reason": "timeout"}


async def quick_exit(
    price_feed: Callable[[], float | Awaitable[float]],
    entry_price: float,
    cfg: Mapping[str, float],
) -> Dict:
    """Convenience wrapper to sell quickly using ``monitor_price``.

    Parameters
    ----------
    price_feed:
        Callable returning the latest price. Can be async.
    entry_price:
        Price at which the asset was bought.
    cfg:
        Configuration mapping. Values ``quick_sell_profit_pct`` and
        ``quick_sell_timeout_sec`` override defaults.
    """

    params = {
        "profit_tp_pct": float(cfg.get("quick_sell_profit_pct", 10)),
        "timeout": float(cfg.get("quick_sell_timeout_sec", 120)),
    }
    return await monitor_price(price_feed, entry_price, params)

