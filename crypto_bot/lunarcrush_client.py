from __future__ import annotations

import os
import requests

from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "lunarcrush.log")


class LunarCrushClient:
    """Simple client for the LunarCrush API."""

    BASE_URL = "https://api.lunarcrush.com/v2"

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None) -> None:
        self.api_key = api_key or os.getenv("LUNARCRUSH_API_KEY")
        self.session = session or requests.Session()

    def get_sentiment(self, symbol: str) -> float:
        """Return sentiment score for ``symbol`` normalised to 0-100."""
        params = {"data": "assets", "symbol": symbol}
        if self.api_key:
            params["key"] = self.api_key
        resp = self.session.get(self.BASE_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
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
