from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

import pandas as pd

from .utils.market_loader import fetch_ohlcv_async
from .strategy import sniper_bot

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import x_semantic_search
except Exception:  # pragma: no cover - library missing
    x_semantic_search = None


async def _process_symbol(exchange: Any, symbol: str, config: dict) -> None:
    """Fetch candles and evaluate using :func:`sniper_bot.generate_signal`."""
    tf = config.get("timeframe", "1m")
    limit = config.get("history_limit", 20)
    try:
        data = await fetch_ohlcv_async(exchange, symbol, timeframe=tf, limit=limit)
    except Exception as exc:  # pragma: no cover - network error
        logger.error("OHLCV fetch failed for %s: %s", symbol, exc)
        return
    if not data:
        logger.info("No OHLCV data for %s", symbol)
        return
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    score, direction = sniper_bot.generate_signal(df)
    logger.info("New listing %s -> %.2f %s", symbol, score, direction)


async def scan_once(exchange: Any, config: dict) -> None:
    """Query X for new listings and evaluate promising symbols."""
    if x_semantic_search is None:  # pragma: no cover - fallback
        logger.warning("x_semantic_search package not available")
        return

    results: Iterable[dict] = x_semantic_search.search("new crypto listing")  # type: ignore[attr-defined]
    vol_mult = float(config.get("volume_multiple", 5.0))
    for item in results:
        base_vol = item.get("baseline_volume") or 0
        quote_vol = item.get("quote_volume") or 0
        symbol = item.get("symbol")
        if symbol and base_vol and quote_vol > vol_mult * base_vol:
            await _process_symbol(exchange, symbol, config)


async def run_scanner(exchange: Any, config: dict) -> None:
    """Run the scanner loop."""
    interval = int(config.get("interval_minutes", 60))
    while True:
        try:
            await scan_once(exchange, config)
        except Exception as exc:  # pragma: no cover - unexpected
            logger.error("Listing scan error: %s", exc, exc_info=True)
        await asyncio.sleep(interval * 60)

