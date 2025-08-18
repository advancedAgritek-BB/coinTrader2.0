from __future__ import annotations

"""Helper utilities for Solana token pricing and OHLCV."""

import aiohttp
import time
from typing import List, Dict, List as _List

from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "meme_sniper.log")


async def fetch_solana_prices(symbols: List[str]) -> Dict[str, float]:
    """Return current prices for ``symbols`` using Jupiter public API."""
    if not symbols:
        return {}
    results: Dict[str, float] = {}
    async with aiohttp.ClientSession() as session:
        for sym in symbols:
            base = sym.split("/")[0]
            try:
                async with session.get(
                    "https://price.jup.ag/v4/price",
                    params={"ids[]": base},
                    timeout=10,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                price = float(data.get("data", {}).get(base, {}).get("price", 0.0))
            except Exception as exc:  # pragma: no cover - network failures
                logger.error("Failed to fetch price for %s: %s", base, exc)
                price = 0.0
            results[sym] = price
    return results


async def fetch_onchain_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
) -> _List[_List[float]]:
    """Return OHLCV candles for an on-chain ``symbol``.

    Attempts to pull historical candles from the Jupiter price API. When the
    request fails or returns no data the function falls back to
    :func:`fetch_solana_prices` and synthesizes a single candle using the latest
    price. The returned structure matches the CCXT OHLCV format ``[timestamp,
    open, high, low, close, volume]`` with timestamps expressed in
    milliseconds.
    """

    base, _, _ = symbol.partition("/")
    url = "https://price.jup.ag/v4/candles"
    params = {"ids[]": base, "interval": timeframe, "limit": limit}
    try:  # pragma: no cover - network usage
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
        candles = data.get("data", {}).get(base, [])
        if candles:
            return [
                [
                    int(c.get("t", 0)) * 1000,
                    float(c.get("o", 0.0)),
                    float(c.get("h", 0.0)),
                    float(c.get("l", 0.0)),
                    float(c.get("c", 0.0)),
                    float(c.get("v", 0.0)),
                ]
                for c in candles
            ]
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("Failed to fetch on-chain OHLCV for %s: %s", symbol, exc)

    # Fallback – synthesize a single candle from the latest price
    try:
        price = (await fetch_solana_prices([symbol])).get(symbol, 0.0)
        if price:
            ts = int(time.time() * 1000)
            return [[ts, price, price, price, price, 0.0]]
    except Exception:
        pass
    return []
