"""Utilities for gauging market sentiment."""

from __future__ import annotations

import os

import aiohttp

from crypto_bot.lunarcrush_client import LunarCrushClient

from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "sentiment.log")

lunar_client = LunarCrushClient()


FNG_URL = "https://api.alternative.me/fng/?limit=1"
SENTIMENT_URL = os.getenv(
    "TWITTER_SENTIMENT_URL", "https://api.example.com/twitter-sentiment"
)


async def fetch_fng_index() -> int:
    """Return the current Fear & Greed index (0-100)."""
    mock = os.getenv("MOCK_FNG_VALUE")
    if mock is not None:
        try:
            return int(mock)
        except ValueError:
            return 50
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(FNG_URL, timeout=5) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if isinstance(data, dict):
                    return int(data.get("data", [{}])[0].get("value", 50))
    except Exception as exc:
        logger.error("Failed to fetch FNG index: %s", exc)
    return 50


async def fetch_twitter_sentiment(
    query: str = "bitcoin", symbol: str | None = None
) -> int:
    """Return sentiment score for ``query`` between 0-100.

    When ``symbol`` is provided and ``LUNARCRUSH_API_KEY`` is set this will
    return LunarCrush sentiment for that symbol instead of calling the
    external Twitter API.
    """
    mock = os.getenv("MOCK_TWITTER_SENTIMENT")
    if mock is not None:
        try:
            return int(mock)
        except ValueError:
            return 50
    if symbol and os.getenv("LUNARCRUSH_API_KEY"):
        return await fetch_lunarcrush_sentiment(symbol)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{SENTIMENT_URL}?q={query}", timeout=5) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if isinstance(data, dict):
                    return int(data.get("score", 50))
    except Exception as exc:
        logger.error("Failed to fetch Twitter sentiment: %s", exc)
    return 50


async def fetch_lunarcrush_sentiment(symbol: str) -> int:
    """Return sentiment score for ``symbol`` using LunarCrush."""
    try:
        return int(await lunar_client.get_sentiment(symbol))
    except Exception as exc:  # pragma: no cover - network failure
        logger.error("Failed to fetch LunarCrush sentiment: %s", exc)
        return 50


async def too_bearish(
    min_fng: int, min_sentiment: int, *, symbol: str | None = None
) -> bool:
    """Return ``True`` when sentiment is below thresholds."""
    fng = await fetch_fng_index()
    if symbol:
        sentiment = await fetch_lunarcrush_sentiment(symbol)
    else:
        sentiment = await fetch_twitter_sentiment()
    logger.info("FNG %s, sentiment %s", fng, sentiment)
    return fng < min_fng or sentiment < min_sentiment


async def boost_factor(
    bull_fng: int, bull_sentiment: int, *, symbol: str | None = None
) -> float:
    """Return a trade size boost factor based on strong sentiment."""
    fng = await fetch_fng_index()
    if symbol:
        sentiment = await fetch_lunarcrush_sentiment(symbol)
    else:
        sentiment = await fetch_twitter_sentiment()
    if fng > bull_fng and sentiment > bull_sentiment:
        factor = 1 + ((fng - bull_fng) + (sentiment - bull_sentiment)) / 200
        logger.info("Applying boost factor %.2f", factor)
        return factor
    return 1.0

