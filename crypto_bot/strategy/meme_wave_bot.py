from __future__ import annotations

import os
from typing import Optional, Tuple

import aiohttp
import pandas as pd

from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor


WEB_SEARCH_URL = os.getenv("WEB_SEARCH_URL", "https://api.example.com/search")


async def web_search(query: str) -> int:
    """Return the number of results for ``query`` using a web API."""
    from urllib.parse import quote_plus

    url = f"{WEB_SEARCH_URL}?q={quote_plus(query)}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return 0
                data = await resp.json()
                count = data.get("count", 0)
                try:
                    return int(count)
                except Exception:
                    return 0
    except Exception:
        return 0


async def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    *,
    symbol: str | None = None,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
) -> Tuple[float, str]:
    """Return a meme-wave trading signal based on momentum and social buzz."""

    if df is None or df.empty:
        return 0.0, "none"

    params = config.get("meme_wave", {}) if config else {}
    lookback = int(params.get("lookback", 20))
    if len(df) < lookback:
        return 0.0, "none"

    recent = df.iloc[-(lookback + 1) :]
    ma = recent["close"].ewm(span=lookback, adjust=False).mean()
    momentum = recent["close"].iloc[-1] - ma.iloc[-1]

    direction = "none"
    if momentum > 0:
        direction = "long"
    elif momentum < 0:
        direction = "short"

    score = abs(momentum) / ma.iloc[-1] if ma.iloc[-1] else 0.0

    query = symbol or params.get("query")
    if query:
        mentions = await web_search(query)
        min_mentions = int(params.get("min_mentions", 0))
        if mentions < min_mentions:
            return 0.0, "none"
        weight = float(params.get("mention_weight", 0.01))
        score += mentions * weight

    if mempool_monitor is not None:
        try:
            fee = mempool_monitor.fetch_priority_fee()
            threshold = float(params.get("fee_threshold", 30))
            if fee >= threshold:
                return 0.0, "none"
        except Exception:
            pass

    score = min(score, 1.0)
    if score > 0:
        score = normalize_score_by_volatility(recent, score)

    return score, direction


class regime_filter:
    """Match trending or volatile regimes."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime in {"trending", "volatile"}
