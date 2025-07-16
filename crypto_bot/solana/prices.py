from __future__ import annotations

"""Helper for fetching Solana token prices."""

import aiohttp
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


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
