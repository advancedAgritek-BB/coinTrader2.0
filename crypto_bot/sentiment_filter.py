from __future__ import annotations

import asyncio
import os
import threading
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

    _CACHE[key] = 50
    return 50


async def _get_lunarcrush_sentiment(symbol: str) -> int:
    """Internal helper to retrieve LunarCrush sentiment for ``symbol``."""
    key = f"lunar:{symbol}"
    if key in _CACHE:
        return _CACHE[key]

    try:
        result = lunar_client.get_sentiment(symbol)
        if asyncio.iscoroutine(result):
            value = int(await result)
        else:
            value = int(result)
    except Exception as exc:  # pragma: no cover - network failure
        logger.error("Failed to fetch LunarCrush sentiment: %s", exc)
        value = 50
    _CACHE[key] = value
    return value


def fetch_lunarcrush_sentiment(symbol: str) -> int:
    """Synchronously return LunarCrush sentiment score for ``symbol``."""
    try:
        return asyncio.run(_get_lunarcrush_sentiment(symbol))
    except RuntimeError:
        result: list[int] = []
        error: list[Exception] = []

        def _runner() -> None:
            try:
                result.append(asyncio.run(_get_lunarcrush_sentiment(symbol)))
            except Exception as exc:  # pragma: no cover - thread failure
                error.append(exc)

        thread = threading.Thread(target=_runner)
        thread.start()
        thread.join()
        if error:
            raise error[0]
        return result[0]


async def fetch_lunarcrush_sentiment_async(symbol: str) -> int:
    """Asynchronously return LunarCrush sentiment score for ``symbol``."""
    return await _get_lunarcrush_sentiment(symbol)


def fetch_twitter_sentiment(
    query: str = "bitcoin", symbol: Optional[str] = None
) -> int | asyncio.Future:
    """Return sentiment score using LunarCrush.

    ``symbol`` takes precedence over ``query``. When ``symbol`` is provided a
    coroutine is returned so callers may ``await`` it or run it via
    :func:`asyncio.run`. A mock value can be supplied via the
    ``MOCK_TWITTER_SENTIMENT`` environment variable which is useful for tests.
    When no LunarCrush API key is available a neutral score of 50 is returned
    and an error is logged.
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

    if symbol is not None:
        if not os.getenv("LUNARCRUSH_API_KEY"):
            logger.error(
                "LUNARCRUSH_API_KEY missing; returning neutral sentiment"
            )

            async def _neutral() -> int:
                return 50

            return _neutral()

        async def _fetch() -> int:
            return await fetch_lunarcrush_sentiment_async(target)

        return _fetch()

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
