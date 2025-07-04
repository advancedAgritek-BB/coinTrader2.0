import asyncio
import logging

from crypto_bot.utils import symbol_utils

class DummyExchange:
    pass


def test_get_filtered_symbols_fallback(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    async def fake_filter_symbols(ex, syms):
        return []

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)
    config = {"symbol": "BTC/USD"}
    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result == ["BTC/USD"]
    assert any("falling back" in r.getMessage() for r in caplog.records)


def test_get_filtered_symbols_caching(monkeypatch):
    calls = []

    async def fake_to_thread(fn, *a):
        return fn(*a)

    async def fake_filter_symbols(ex, syms):
        calls.append(True)
        return ["ETH/USD"]

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {
        "symbol": "ETH/USD",
        "symbols": ["ETH/USD"],
        "symbol_refresh_minutes": 1,
    }

    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    result1 = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    result2 = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))

    assert result1 == ["ETH/USD"]
    assert result2 == ["ETH/USD"]
    assert len(calls) == 1

    symbol_utils._last_refresh -= 61

    result3 = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result3 == ["ETH/USD"]
    assert len(calls) == 2
