"""Utilities for gauging market sentiment."""

from __future__ import annotations

import os

from crypto_bot.lunarcrush_client import LunarCrushClient

from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "sentiment.log")

lunar_client = LunarCrushClient()






def fetch_lunarcrush_sentiment(symbol: str) -> int:
    """Return sentiment score for ``symbol`` using LunarCrush."""
    try:
        return int(lunar_client.get_sentiment(symbol))
    except Exception as exc:  # pragma: no cover - network failure
        logger.error("Failed to fetch LunarCrush sentiment: %s", exc)
        return 50


def too_bearish(min_sentiment: int, *, symbol: str | None = None) -> bool:
    """Return ``True`` when LunarCrush sentiment is below ``min_sentiment``."""
    if symbol:
        sentiment = fetch_lunarcrush_sentiment(symbol)
    else:
        sentiment = fetch_lunarcrush_sentiment("BTC")
    logger.info("Sentiment %s", sentiment)
    return sentiment < min_sentiment


def boost_factor(bull_sentiment: int, *, symbol: str | None = None) -> float:
    """Return a trade size boost factor based on strong LunarCrush sentiment."""
    if symbol:
        sentiment = fetch_lunarcrush_sentiment(symbol)
    else:
        sentiment = fetch_lunarcrush_sentiment("BTC")
    if sentiment > bull_sentiment:
        factor = 1 + (sentiment - bull_sentiment) / 100
        logger.info("Applying boost factor %.2f", factor)
        return factor
    return 1.0

