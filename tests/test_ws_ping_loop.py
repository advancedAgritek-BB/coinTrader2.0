import asyncio
import pytest

import crypto_bot.main as main

class DummyExchange:
    def __init__(self):
        self.clients = {"a": object(), "b": object()}
        self.pings = []

    async def ping(self, client=None):
        self.pings.append(client)

class StopLoop(Exception):
    pass


def test_ws_ping_loop_calls_ping_with_client(monkeypatch):
    ex = DummyExchange()

    calls = {"sleep": 0}

    async def fake_sleep(_):
        calls["sleep"] += 1
        if calls["sleep"] >= 2:
            raise StopLoop

    monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)

    with pytest.raises(StopLoop):
        asyncio.run(main._ws_ping_loop(ex, 0))

    assert ex.pings == list(ex.clients.values())


def test_ws_ping_loop_skips_when_no_clients(monkeypatch):
    ex = DummyExchange()
    ex.clients = {}

    calls = {"sleep": 0}

    async def fake_sleep(_):
        calls["sleep"] += 1
        if calls["sleep"] >= 2:
            raise StopLoop

    monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)

    with pytest.raises(StopLoop):
        asyncio.run(main._ws_ping_loop(ex, 0))

    assert ex.pings == []
