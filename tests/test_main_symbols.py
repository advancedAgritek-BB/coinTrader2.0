import asyncio
import logging

from crypto_bot.utils import symbol_utils, market_loader


class DummyExchange:
    markets = {
        "BTC/USD": {"quote": "USD", "quoteVolume": 1_000_000},
        "ETH/USD": {"quote": "USD", "quoteVolume": 1_000_000},
        "SOL/USDC": {"quote": "USDC", "quoteVolume": 1_000_000},
    }

    def list_markets(self):
        return self.markets


def test_get_filtered_symbols_fallback(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    calls = []

    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    async def fake_filter_symbols(ex, syms, cfg):
        calls.append(syms)
        if len(calls) == 1:
            return [], []
        return [("BTC/USD", 1.0)], []

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbol": "BTC/USD"}
    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result[0] == [("BTC/USD", 0.0)]
    assert result == ([("BTC/USD", 0.0)], [])
    assert calls == [config.get("symbols", [config.get("symbol")]), ["BTC/USD"]]
    assert any("falling back" in r.getMessage() for r in caplog.records)


def test_get_filtered_symbols_fallback_excluded(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    async def fake_filter_symbols(ex, syms, cfg):
        return [], []

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbol": "BTC/USD", "excluded_symbols": ["BTC/USD"]}
    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result[0] == []
    assert result == ([], [])
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
        return [], []

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbol": "BTC/USD"}
    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result[0] == []
    assert result == ([], [])
    assert calls == [config.get("symbols", [config.get("symbol")]), ["BTC/USD"]]
    assert any("volume requirements" in r.getMessage() for r in caplog.records)
    assert any("No symbols met volume/spread requirements" in r.getMessage() for r in caplog.records)
    assert symbol_utils._cached_symbols is None
    assert symbol_utils._last_refresh == 0.0


def test_get_filtered_symbols_caching(monkeypatch):
    calls = []

    async def fake_filter_symbols(ex, syms, cfg):
        calls.append(True)
        return [("ETH/USD", 1.0)], []

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

    assert result1[0] == [("ETH/USD", 1.0)]
    assert result2[0] == [("ETH/USD", 1.0)]
    assert result1 == ([("ETH/USD", 1.0)], [])
    assert result2 == ([("ETH/USD", 1.0)], [])
    assert len(calls) == 1

    symbol_utils._last_refresh -= 61

    result3 = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result3[0] == [("ETH/USD", 1.0)]
    assert result3 == ([("ETH/USD", 1.0)], [])
    assert len(calls) == 2


def test_get_filtered_symbols_basic(monkeypatch):
    async def fake_filter_symbols(ex, syms, cfg):
        return [(syms[0], 0.5)], []

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbol": "ETH/USD", "symbols": ["ETH/USD"]}
    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result[0] == [("ETH/USD", 0.5)]
    assert result == ([("ETH/USD", 0.5)], [])


def test_get_filtered_symbols_invalid_usdc_token(monkeypatch):
    calls: list[list[str]] = []

    async def fake_filter_symbols(ex, syms, cfg):
        calls.append(syms)
        return [], []

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
        return [(s, 1.0) for s in syms], []

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbols": ["BAD/USDC", "ETH/USD"]}
    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result[0] == [("ETH/USD", 1.0)]
    assert result == ([("ETH/USD", 1.0)], [])
    assert calls == [["ETH/USD"]]
    assert not any("invalid USDC" in r.getMessage() for r in caplog.records)


def test_get_filtered_symbols_skip(monkeypatch):
    async def fake_filter_symbols(ex, syms, cfg):
        raise AssertionError("filter_symbols should not be called")

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbols": ["BTC/USD", "ETH/USD"], "skip_symbol_filters": True}
    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result[0] == [("BTC/USD", 0.0), ("ETH/USD", 0.0)]
    assert result == ([("BTC/USD", 0.0), ("ETH/USD", 0.0)], [])


def test_get_filtered_symbols_spread_filter(monkeypatch):
    async def fake_filter_symbols(_ex, syms, cfg):
        spreads = {"BTC/USD": 0.5, "ETH/USD": 2.0}
        max_spread = cfg.get("symbol_filter", {}).get("max_spread_pct", float("inf"))
        return ([(s, spreads[s]) for s in syms if spreads[s] <= max_spread], [])

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {
        "symbols": ["BTC/USD", "ETH/USD"],
        "symbol_filter": {"max_spread_pct": 1.0},
    }
    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))
    assert result == ([("BTC/USD", 0.5)], [])
    assert all(spread <= 1.0 for _, spread in result[0])


def test_get_filtered_symbols_valid_sol(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    monkeypatch.setattr(
        symbol_utils,
        "TOKEN_MINTS",
        {"SOL": "So11111111111111111111111111111111111111112"},
        raising=False,
    )
    monkeypatch.setattr(market_loader, "_is_valid_base_token", lambda t: True)
    monkeypatch.setattr(symbol_utils, "_is_valid_base_token", lambda t: True)

    async def fake_filter_symbols(_ex, syms, _cfg):
        return [("SOL/USDC", 1.0)], []

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbols": ["SOL/USDC"]}
    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyExchange(), config))

    assert result[0] == [("SOL/USDC", 1.0)]
    assert result == ([("SOL/USDC", 1.0)], [])
    assert not any("Dropping invalid USDC pair" in r.getMessage() for r in caplog.records)


def test_get_filtered_symbols_onchain_pair(monkeypatch):
    monkeypatch.setattr(
        symbol_utils,
        "TOKEN_MINTS",
        {"AAA": "mint"},
        raising=False,
    )
    monkeypatch.setattr(symbol_utils, "_is_valid_base_token", lambda t: True)

    async def fake_filter_symbols(_ex, syms, _cfg):
        return [(s, 1.0) for s in syms], []

    class DummyEx:
        markets = {"ETH/USD": {}}

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter_symbols)

    config = {"symbols": ["AAA/USDC", "ETH/USD"]}
    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    result = asyncio.run(symbol_utils.get_filtered_symbols(DummyEx(), config))

    assert result == ([("ETH/USD", 1.0)], [])
