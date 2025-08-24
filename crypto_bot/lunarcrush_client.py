"""Async client for the LunarCrush API."""

from __future__ import annotations

import os
import asyncio

import aiohttp

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.http_client import get_session


logger = setup_logger(__name__, LOG_DIR / "lunarcrush.log")


class LunarCrushClient:
    """Simple client for the LunarCrush API."""

    BASE_URL = "https://api.lunarcrush.com/v2"

    def __init__(
        self,
        api_key: str | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("LUNARCRUSH_API_KEY")
        self.session = session

    async def get_sentiment(self, symbol: str) -> float:
        """Return sentiment score for ``symbol`` normalised to 0-100."""
        params = {"data": "assets", "symbol": symbol}
        if self.api_key:
            params["key"] = self.api_key

        session = self.session or get_session()
        return await self._fetch(session, params)

    def get_sentiment_sync(self, symbol: str) -> float:
        """Return sentiment score for ``symbol`` using ``asyncio.run``.

        This provides a synchronous interface for environments where
        running an event loop is inconvenient.
        """
        return asyncio.run(self.get_sentiment(symbol))

    async def _fetch(
        self, session: aiohttp.ClientSession, params: dict[str, str]
    ) -> float:
        async with session.get(self.BASE_URL, params=params, timeout=5) as resp:
            resp.raise_for_status()
            data = await resp.json()
        value = 0.0
        try:
            value = float(data.get("data", [{}])[0].get("sentiment", 0.0))
        except Exception:
            logger.error("Malformed LunarCrush response: %s", data)
            return 50.0
        # Sentiment values range roughly -5..5, normalise to 0-100
        score = 50.0 + value * 10.0
        return max(0.0, min(100.0, score))

__all__ = ["LunarCrushClient"]
