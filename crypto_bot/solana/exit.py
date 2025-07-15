"""Position exit helpers."""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Mapping, Dict


async def monitor_price(
    price_feed: Callable[[], float],
    entry_price: float,
    cfg: Mapping[str, float],
) -> Dict:
    """Monitor price until exit conditions are met."""

    take_profit = float(cfg.get("take_profit_pct", 0))
    trailing = float(cfg.get("trailing_stop_pct", 0))
    timeout = float(cfg.get("timeout", 60))
    poll = float(cfg.get("poll_interval", 1))
    profit_tp = float(cfg.get("profit_tp_pct", 0.2))

    peak = entry_price
    start = time.time()
    price = entry_price
    while time.time() - start < timeout:
        await asyncio.sleep(poll)
        price = price_feed()
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
