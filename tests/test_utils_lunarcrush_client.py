import asyncio
import pytest

from crypto_bot.utils.lunarcrush_client import LunarCrushClient

class DummyResp:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def json(self):
        return {}

    def raise_for_status(self):
        pass

class DummySession:
    def __init__(self):
        self.last_params = None
        self.closed = False
    def get(self, url, params=None, timeout=10):
        self.last_params = params
        return DummyResp()

    async def close(self):
        self.closed = True

def test_env_key_used(monkeypatch):
    session = DummySession()
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "envkey")
    async def fake_get_session(self):
        return session
    monkeypatch.setattr(LunarCrushClient, "_get_session", fake_get_session)

    client = LunarCrushClient()

    async def _run():
        await client.request("assets", {"symbol": "BTC"})

    asyncio.run(_run())

    assert session.last_params["key"] == "envkey"
