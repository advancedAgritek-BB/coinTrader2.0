import asyncio
import sys
import types
import pytest

telegram_stub = types.ModuleType("crypto_bot.utils.telegram")
telegram_stub.TelegramNotifier = object
telegram_stub.send_test_message = lambda *a, **k: None
telegram_stub.send_message = lambda *a, **k: None
telegram_stub.send_message_sync = lambda *a, **k: None
sys.modules["crypto_bot.utils.telegram"] = telegram_stub

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


def test_ws_ping_loop_sleeps_five_seconds(monkeypatch):
    ex = DummyExchange()

    intervals = []

    async def fake_sleep(interval):
        intervals.append(interval)
        raise StopLoop

    monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)

    with pytest.raises(StopLoop):
        asyncio.run(main._ws_ping_loop(ex, 5))

    assert intervals == [5]
