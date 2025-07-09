import asyncio

import pytest

from crypto_bot.solana import api_helpers


class DummyWS:
    pass


class DummyResp:
    def __init__(self, data):
        self.data = data

    async def json(self):
        return self.data

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummySession:
    def __init__(self):
        self.ws_url = None
        self.http_url = None

    async def ws_connect(self, url):
        self.ws_url = url
        return DummyWS()

    def get(self, url, headers=None, timeout=10):
        self.http_url = url
        return DummyResp({"bundle": "ok"})

    async def close(self):
        pass


def test_connect_helius_ws(monkeypatch):
    session = DummySession()
    monkeypatch.setattr(api_helpers, "aiohttp", type("M", (), {"ClientSession": lambda: session}))
    ws = asyncio.run(api_helpers.connect_helius_ws("k"))
    assert isinstance(ws, DummyWS)
    assert session.ws_url.endswith("k")


def test_fetch_jito_bundle(monkeypatch):
    session = DummySession()
    monkeypatch.setattr(api_helpers, "aiohttp", type("M", (), {"ClientSession": lambda: session}))
    data = asyncio.run(api_helpers.fetch_jito_bundle("123", "key"))
    assert data == {"bundle": "ok"}
    assert session.http_url.endswith("/123")
