import asyncio
import pytest

from crypto_bot.utils.lunarcrush_client import LunarCrushClient
from tests.dummy_aiohttp import DummySession


def test_env_key_used(monkeypatch):
    session = DummySession({})
    monkeypatch.setenv("LUNARCRUSH_API_KEY", "envkey")

    async def fake_get_session(self):
        return session

    monkeypatch.setattr(LunarCrushClient, "_get_session", fake_get_session)

    client = LunarCrushClient()

    async def _run():
        await client.request("assets", {"symbol": "BTC"})

    asyncio.run(_run())

    assert session.params["key"] == "envkey"
