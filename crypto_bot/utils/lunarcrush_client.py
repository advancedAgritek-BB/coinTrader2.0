from __future__ import annotations

import asyncio
import os
from typing import Any, Mapping

import aiohttp

from .logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "lunarcrush_client.log")


class LunarCrushClient:
    """Simple async wrapper around the LunarCrush API."""

    BASE_URL = "https://api.lunarcrush.com/v2"

    def __init__(self, api_key: str | None = None) -> None:
        """Create client using ``api_key`` or ``LUNARCRUSH_API_KEY`` env var."""
        self.api_key = api_key or os.getenv("LUNARCRUSH_API_KEY")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def request(
        self, endpoint: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any] | None:
        """Return parsed JSON from ``endpoint`` with ``params``."""
        url = f"{self.BASE_URL}/{endpoint}"
        p = dict(params or {})
        p["key"] = self.api_key
        session = await self._get_session()
        try:
            async with session.get(url, params=p, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            logger.error("LunarCrush request failed for %s: %s", url, exc)
            return None

    async def get_assets(self, symbol: str) -> Mapping[str, Any] | None:
        """Return asset data for ``symbol``."""
        data = await self.request("assets", {"symbol": symbol})
        if isinstance(data, dict):
            return data.get("data")
        return None

    async def get_market_pairs(self, symbol: str) -> Mapping[str, Any] | None:
        """Return market pairs for ``symbol``."""
        data = await self.request("market-pairs", {"symbol": symbol})
        if isinstance(data, dict):
            return data.get("data")
        return None
