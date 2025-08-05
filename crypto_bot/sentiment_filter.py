"""Sentiment helpers relying solely on LunarCrush data.

This module previously fell back to a Twitter sentiment HTTP API. That
behaviour has been removed: all sentiment lookups are now served via the
`LunarCrushClient`.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import requests
from cachetools import TTLCache

from crypto_bot.lunarcrush_client import LunarCrushClient
from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "sentiment.log")

# Re-used client for sentiment queries
lunar_client = LunarCrushClient()

# Cache for all sentiment lookups (5 minutes)
_CACHE: TTLCache[str, int] = TTLCache(maxsize=128, ttl=300)


FNG_URL = "https://api.alternative.me/fng/?limit=1"


def fetch_fng_index() -> int:
    """Return the current Fear & Greed index (0-100)."""

    mock = os.getenv("MOCK_FNG_VALUE")
    if mock is not None:
        try:
            return int(mock)
        except ValueError:
            return 50

    key = "fng"
    if key in _CACHE:
        return _CACHE[key]

    try:
        resp = requests.get(FNG_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            value = int(data.get("data", [{}])[0].get("value", 50))
            _CACHE[key] = value
            return value
    except Exception as exc:  # pragma: no cover - network failure
        logger.error("Failed to fetch FNG index: %s", exc)

    return 50


def fetch_lunarcrush_sentiment(symbol: str) -> int:
    """Synchronously return LunarCrush sentiment score for ``symbol``."""

    key = f"lunar:{symbol}"
    if key in _CACHE:
        return _CACHE[key]

    try:
        value = int(asyncio.run(lunar_client.get_sentiment(symbol)))
        _CACHE[key] = value
        return value
    except Exception as exc:  # pragma: no cover - network failure
        logger.error("Failed to fetch LunarCrush sentiment: %s", exc)
        return 50


async def fetch_lunarcrush_sentiment_async(symbol: str) -> int:
    """Asynchronously return LunarCrush sentiment score for ``symbol``."""

    key = f"lunar:{symbol}"
    if key in _CACHE:
        return _CACHE[key]

    try:
        value = int(await lunar_client.get_sentiment(symbol))
        _CACHE[key] = value
        return value
    except Exception as exc:  # pragma: no cover - network failure
        logger.error("Failed to fetch LunarCrush sentiment: %s", exc)
        return 50


def fetch_twitter_sentiment(
    query: str = "bitcoin", symbol: Optional[str] = None
) -> int:
    """Return sentiment score using LunarCrush.

    ``symbol`` takes precedence over ``query``. A mock value can be supplied via
    the ``MOCK_TWITTER_SENTIMENT`` environment variable which is useful for
    tests. When no LunarCrush API key is available a neutral score of 50 is
    returned and an error is logged.
    """

    mock = os.getenv("MOCK_TWITTER_SENTIMENT")
    if mock is not None:
        try:
            return int(mock)
        except ValueError:
            return 50

    target = symbol or query
    if not target:
        return 50

    if not os.getenv("LUNARCRUSH_API_KEY"):
        logger.error("LUNARCRUSH_API_KEY missing; returning neutral sentiment")
        return 50

    return fetch_lunarcrush_sentiment(target)


async def fetch_twitter_sentiment_async(
    query: str = "bitcoin", symbol: Optional[str] = None
) -> int:
    """Asynchronously return sentiment score using LunarCrush."""

    mock = os.getenv("MOCK_TWITTER_SENTIMENT")
    if mock is not None:
        try:
            return int(mock)
        except ValueError:
            return 50

    target = symbol or query
    if not target:
        return 50

    if not os.getenv("LUNARCRUSH_API_KEY"):
        logger.error("LUNARCRUSH_API_KEY missing; returning neutral sentiment")
        return 50

    return await fetch_lunarcrush_sentiment_async(target)


async def too_bearish(
    min_fng: int, min_sentiment: int, *, symbol: Optional[str] = None
) -> bool:
    """Return ``True`` when sentiment is below thresholds."""

    fng = fetch_fng_index()
    if symbol:
        sentiment = await fetch_lunarcrush_sentiment_async(symbol)
    else:
        sentiment = await fetch_twitter_sentiment_async()
    logger.info("FNG %s, sentiment %s", fng, sentiment)
    return fng < min_fng or sentiment < min_sentiment


async def boost_factor(
    bull_fng: int, bull_sentiment: int, *, symbol: Optional[str] = None
) -> float:
    """Return a trade size boost factor based on strong sentiment."""

    fng = fetch_fng_index()
    if symbol:
        sentiment = await fetch_lunarcrush_sentiment_async(symbol)
    else:
        sentiment = await fetch_twitter_sentiment_async()
    if fng > bull_fng and sentiment > bull_sentiment:
        factor = 1 + ((fng - bull_fng) + (sentiment - bull_sentiment)) / 200
        logger.info("Applying boost factor %.2f", factor)
        return factor
    return 1.0


__all__ = [
    "boost_factor",
    "fetch_fng_index",
    "fetch_lunarcrush_sentiment",
    "fetch_lunarcrush_sentiment_async",
    "fetch_twitter_sentiment",
    "fetch_twitter_sentiment_async",
    "too_bearish",
]

