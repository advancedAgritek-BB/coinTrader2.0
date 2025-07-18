import asyncio
import logging

from crypto_bot.utils import symbol_utils

class DummyExchange:
    pass


def test_get_filtered_symbols_fallback(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    calls = []

    async def fake_filter_symbols(ex, syms, cfg):
        calls.append(syms)
        if len(calls) == 1:
            return []
        return [("BTC/USD", 1.0)]

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbol": "BTC/USD"}
    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result == [("BTC/USD", 0.0)]
    assert calls == [config.get("symbols", [config.get("symbol")]), ["BTC/USD"]]
    assert any("falling back" in r.getMessage() for r in caplog.records)


def test_get_filtered_symbols_fallback_excluded(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    async def fake_filter_symbols(ex, syms, cfg):
        return []

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbol": "BTC/USD", "excluded_symbols": ["BTC/USD"]}
    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result == []
    assert any("excluded" in r.getMessage() for r in caplog.records)
    assert any("No symbols met volume/spread requirements" in r.getMessage() for r in caplog.records)
    assert symbol_utils._cached_symbols is None
    assert symbol_utils._last_refresh == 0.0


def test_get_filtered_symbols_fallback_volume_fail(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    calls = []

    async def fake_filter_symbols(ex, syms, cfg):
        calls.append(syms)
        return []

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbol": "BTC/USD"}
    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result == []
    assert calls == [config.get("symbols", [config.get("symbol")]), ["BTC/USD"]]
    assert any("volume requirements" in r.getMessage() for r in caplog.records)
    assert any("No symbols met volume/spread requirements" in r.getMessage() for r in caplog.records)
    assert symbol_utils._cached_symbols is None
    assert symbol_utils._last_refresh == 0.0


def test_get_filtered_symbols_caching(monkeypatch):
    calls = []

    async def fake_to_thread(fn, *a):
        return fn(*a)

    async def fake_filter_symbols(ex, syms, cfg):
        calls.append(True)
        return [("ETH/USD", 1.0)]

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

    assert result1 == [("ETH/USD", 1.0)]
    assert result2 == [("ETH/USD", 1.0)]
    assert len(calls) == 1

    symbol_utils._last_refresh -= 61

    result3 = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result3 == [("ETH/USD", 1.0)]
    assert len(calls) == 2


def test_get_filtered_symbols_basic(monkeypatch):
    async def fake_filter_symbols(ex, syms, cfg):
        return [(syms[0], 0.5)]

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbol": "ETH/USD", "symbols": ["ETH/USD"]}
    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result == [("ETH/USD", 0.5)]


def test_get_filtered_symbols_invalid_usdc_token(monkeypatch):
    calls: list[list[str]] = []

    async def fake_filter_symbols(ex, syms, cfg):
        calls.append(syms)
        return []

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {
        "symbol": "BTC/USD",
        "symbols": ["ALGO/USDC", "BTC/USD"],
    }

    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))

    assert calls and calls[0] == ["BTC/USD"]
def test_get_filtered_symbols_invalid_usdc(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    calls = []

    async def fake_filter_symbols(ex, syms, cfg):
        calls.append(syms)
        return [(s, 1.0) for s in syms]

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbols": ["BAD/USDC", "ETH/USD"]}
    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result == [("ETH/USD", 1.0)]
    assert calls == [["ETH/USD"]]
    assert any("invalid USDC" in r.getMessage() for r in caplog.records)


def test_get_filtered_symbols_skip(monkeypatch):
    async def fake_filter_symbols(ex, syms, cfg):
        raise AssertionError("filter_symbols should not be called")

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbols": ["BTC/USD", "ETH/USD"], "skip_symbol_filters": True}
    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result == [("BTC/USD", 0.0), ("ETH/USD", 0.0)]
