import asyncio
import logging
import sys
import types
import pytest

sys.modules.setdefault("gspread", types.ModuleType("gspread"))
sys.modules.setdefault(
    "oauth2client.service_account",
    types.SimpleNamespace(ServiceAccountCredentials=object),
)
sys.modules.setdefault("ta", types.ModuleType("ta"))
sys.modules.setdefault("scipy", types.ModuleType("scipy"))
sys.modules.setdefault(
    "scipy.stats",
    types.SimpleNamespace(pearsonr=lambda *_a, **_k: 0),
)
sys.modules.setdefault("rich.console", types.SimpleNamespace(Console=object))
sys.modules.setdefault("rich.table", types.SimpleNamespace(Table=object))

import crypto_bot.main as main


class DummyExchangeNoClose:
    async def watch_ticker(self, symbol):
        raise asyncio.TimeoutError("timeout")


class DummyExchangeBadClose:
    async def watch_ticker(self, symbol):
        raise asyncio.TimeoutError("timeout")

    async def close(self):
        raise AttributeError("boom")


async def dummy_ping_loop(*_a, **_k):
    return None


def test_watch_position_logs_missing_close(monkeypatch, caplog):
    positions = {"BTC/USD": {}}
    ex = DummyExchangeNoClose()
    config = {"use_websocket": True, "position_poll_interval": 0, "ws_ping_interval": 0}

    def fake_get_exchange(cfg):
        positions.pop("BTC/USD", None)
        return ex, None

    monkeypatch.setattr(main, "get_exchange", fake_get_exchange)
    monkeypatch.setattr(main, "_ws_ping_loop", dummy_ping_loop)

    caplog.set_level(logging.WARNING)
    asyncio.run(main._watch_position("BTC/USD", ex, positions, None, config, {}))

    assert any("Exchange missing close method" in r.getMessage() for r in caplog.records)


def test_watch_position_logs_close_error(monkeypatch, caplog):
    positions = {"BTC/USD": {}}
    ex = DummyExchangeBadClose()
    config = {"use_websocket": True, "position_poll_interval": 0, "ws_ping_interval": 0}

    def fake_get_exchange(cfg):
        positions.pop("BTC/USD", None)
        return ex, None

    monkeypatch.setattr(main, "get_exchange", fake_get_exchange)
    monkeypatch.setattr(main, "_ws_ping_loop", dummy_ping_loop)

    caplog.set_level(logging.ERROR)
    asyncio.run(main._watch_position("BTC/USD", ex, positions, None, config, {}))

    assert any("Exchange close failed" in r.getMessage() for r in caplog.records)
