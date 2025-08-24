import aiohttp
import pytest

import crypto_bot.lunarcrush_client as lc
from crypto_bot.lunarcrush_client import LunarCrushClient


class DummyResponse:
    def __init__(self, data):
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def json(self):
        return self._data

    def raise_for_status(self):
        pass


class DummySession:
    def __init__(self, data):
        self.data = data
        self.last_params = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, url, params=None, timeout=5):
        self.last_params = params
        return DummyResponse(self.data)


@pytest.mark.asyncio
async def test_get_sentiment(monkeypatch):
    session = DummySession({"data": [{"sentiment": 4}]})
    monkeypatch.setattr(lc, "get_session", lambda: session)
    client = LunarCrushClient()
    score = await client.get_sentiment("BTC")
    assert session.last_params["symbol"] == "BTC"
    assert score == 90.0
