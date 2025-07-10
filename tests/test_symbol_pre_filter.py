import asyncio
import json
import pandas as pd
import pytest
import crypto_bot.utils.symbol_scoring as sc

from crypto_bot.utils.symbol_pre_filter import filter_symbols, has_enough_history

CONFIG = {
    "symbol_filter": {
        "min_volume_usd": 50000,
        "max_spread_pct": 2.0,
        "correlation_max_pairs": 10,
    },
    "symbol_score_weights": {"volume": 1, "change": 0, "spread": 0, "age": 0, "latency": 0},
    "max_vol": 100000,
    "min_symbol_score": 0.0,
}

class DummyExchange:
    markets_by_id = {
        "XETHZUSD": {"symbol": "ETH/USD"},
        "XXBTZUSD": {"symbol": "BTC/USD"},
    }


async def fake_fetch(_):
    return {
        "result": {
            "XETHZUSD": {
                "a": ["101", "1", "1"],
                "b": ["100", "1", "1"],
                "c": ["101", "0.5"],
                "v": ["800", "800"],
                "p": ["100", "100"],
                "o": "99",
            },
            "XXBTZUSD": {
                "a": ["51", "1", "1"],
                "b": ["50", "1", "1"],
                "c": ["51", "1"],
                "v": ["600", "600"],
                "p": ["100", "100"],
                "o": "49",
            },
        }
    }


def test_filter_symbols(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )
    symbols = asyncio.run(
        filter_symbols(DummyExchange(), ["ETH/USD", "BTC/USD"], CONFIG)
    )
    assert symbols == [("BTC/USD", 0.6)]


class DummyExchangeList:
    # markets_by_id values may be lists of market dictionaries
    markets_by_id = {"XETHZUSD": [{"symbol": "ETH/USD"}]}


def test_filter_symbols_handles_list_values(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )
    symbols = asyncio.run(filter_symbols(DummyExchangeList(), ["ETH/USD"], CONFIG))
    assert symbols == [("ETH/USD", 0.8)]
class EmptyExchange:
    def __init__(self):
        self.markets_by_id = {}
        self.called = False

    def load_markets(self):
        self.called = True
        self.markets_by_id = {"XETHZUSD": {"symbol": "ETH/USD"}}


def test_load_markets_when_missing(monkeypatch):
    ex = EmptyExchange()
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )
    symbols = asyncio.run(filter_symbols(ex, ["ETH/USD"], CONFIG))
    assert ex.called is True
    assert symbols == [("ETH/USD", 0.8)]


def test_non_dict_market_entry(monkeypatch):
    class BadExchange:
        markets_by_id = {"XETHZUSD": ["ETH/USD"]}

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )
    symbols = asyncio.run(filter_symbols(BadExchange(), ["ETH/USD"], CONFIG))
    assert symbols == [("XETHZUSD", 0.8)]


def test_multiple_batches(monkeypatch):
    calls = []

    async def fake_fetch_multi(pairs_param):
        pairs_list = list(pairs_param)
        combined = {"result": {}}
        for i in range(0, len(pairs_list), 20):
            chunk = pairs_list[i : i + 20]
            calls.append(chunk)
            ticker = {
                "a": ["101", "1", "1"],
                "b": ["100", "1", "1"],
                "c": ["101", "0.5"],
                "v": ["600", "600"],
                "p": ["100", "100"],
                "o": "99",
            }
            combined["result"].update({p: ticker for p in chunk})
        return combined

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch_multi
    )

    pairs = [f"PAIR{i}" for i in range(25)]

    class DummyEx:
        markets_by_id = {p: {"symbol": p} for p in pairs}

    symbols = asyncio.run(filter_symbols(DummyEx(), pairs, CONFIG))

    assert len(symbols) == 25
    assert len(calls) == 2


async def mock_fetch_history(exchange, symbol, timeframe="1d", limit=30, **_):
    return [[i * 86400000, 0, 0, 0, 0, 0] for i in range(limit)]


def test_has_enough_history_true(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter.fetch_ohlcv_async",
        mock_fetch_history,
    )
    assert asyncio.run(has_enough_history(None, "BTC/USD", days=10))


async def mock_fetch_history_short(exchange, symbol, timeframe="1d", limit=30, **_):
    return [[i * 86400000, 0, 0, 0, 0, 0] for i in range(5)]


def test_has_enough_history_false(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter.fetch_ohlcv_async",
        mock_fetch_history_short,
    )
    assert not asyncio.run(has_enough_history(None, "BTC/USD", days=10))


async def mock_fetch_history_error(exchange, symbol, timeframe="1d", limit=30, **_):
    return asyncio.TimeoutError()


def test_has_enough_history_error(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter.fetch_ohlcv_async",
        mock_fetch_history_error,
    )
    assert not asyncio.run(has_enough_history(None, "BTC/USD", days=10))
async def mock_fetch_history_exception(exchange, symbol, timeframe="1d", limit=30, **_):
    import ccxt
    return ccxt.RequestTimeout("timeout")


def test_has_enough_history_exception(monkeypatch, caplog):
    caplog.set_level("WARNING")
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter.fetch_ohlcv_async",
        mock_fetch_history_exception,
    )
    assert not asyncio.run(has_enough_history(None, "BTC/USD", days=10))
    assert any("returned exception" in r.getMessage() for r in caplog.records)
def test_filter_symbols_sorted_by_score(monkeypatch):
    async def fake_fetch_sorted(_):
        return {
            "result": {
                "XETHZUSD": {
                    "a": ["101", "1", "1"],
                    "b": ["100", "1", "1"],
                    "c": ["101", "0.5"],
                    "v": ["600", "600"],
                    "p": ["100", "100"],
                    "o": "99",
                },
                "XXBTZUSD": {
                    "a": ["51", "1", "1"],
                    "b": ["50", "1", "1"],
                    "c": ["51", "1"],
                    "v": ["800", "800"],
                    "p": ["100", "100"],
                    "o": "49",
                },
            }
        }

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch_sorted
    )

    cfg = {
        **CONFIG,
        "symbol_filter": {"min_volume_usd": 50000, "change_pct_percentile": 0, "max_spread_pct": 2.0},
    }
    symbols = asyncio.run(
        filter_symbols(DummyExchange(), ["ETH/USD", "BTC/USD"], cfg)
    )

    assert symbols == [("BTC/USD", 0.8), ("ETH/USD", 0.6)]


def test_filter_symbols_min_score(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )

    cfg = {
        **CONFIG,
        "min_symbol_score": 0.7,
        "symbol_filter": {"min_volume_usd": 50000, "change_pct_percentile": 0},
    }
    symbols = asyncio.run(filter_symbols(DummyExchange(), ["ETH/USD", "BTC/USD"], cfg))
    assert symbols == [("ETH/USD", 0.8)]
class HistoryExchange:
    def __init__(self, candles: int):
        self.markets_by_id = {"XETHZUSD": {"symbol": "ETH/USD"}}
        self.candles = candles

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[i * 3600, 0, 0, 0, 0, 0] for i in range(min(self.candles, limit))]


def test_filter_symbols_min_age_skips(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )
    cfg = {**CONFIG, "min_symbol_age_days": 2}
    ex = HistoryExchange(24)
    symbols = asyncio.run(filter_symbols(ex, ["ETH/USD"], cfg))
    assert symbols == []


def test_filter_symbols_min_age_allows(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )
    cfg = {**CONFIG, "min_symbol_age_days": 2}
    ex = HistoryExchange(48)
    symbols = asyncio.run(filter_symbols(ex, ["ETH/USD"], cfg))
    assert symbols == [("ETH/USD", 0.8)]


def test_filter_symbols_correlation(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async",
        fake_fetch,
    )
    df1 = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    df2 = pd.DataFrame({"close": [2, 4, 6, 8, 10]})
    cache = {"ETH/USD": df1, "BTC/USD": df2}

    cfg = {
        **CONFIG,
        "symbol_filter": {"min_volume_usd": 50000, "max_spread_pct": 2.0, "change_pct_percentile": 0},
    }
    symbols = asyncio.run(
        filter_symbols(DummyExchange(), ["ETH/USD", "BTC/USD"], cfg, df_cache=cache)
    )

    assert symbols == [("ETH/USD", 0.8)]
async def fake_fetch_wide_spread(_):
    return {
        "result": {
            "XETHZUSD": {
                "a": ["102", "1", "1"],
                "b": ["98", "1", "1"],
                "c": ["100", "0.5"],
                "v": ["800", "800"],
                "p": ["100", "100"],
                "o": "98",
            }
        }
    }


def test_filter_symbols_spread(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async",
        fake_fetch_wide_spread,
    )
    cfg = {"symbol_filter": {"min_volume_usd": 50000, "max_spread_pct": 1.0}}
    symbols = asyncio.run(filter_symbols(DummyExchange(), ["ETH/USD"], cfg))
    assert symbols == []
def test_percentile_selects_top_movers(monkeypatch):
    async def fake_fetch_pct(_):
        result = {}
        for i in range(1, 11):
            price = 100 + i
            result[f"PAIR{i}"] = {
                "a": [str(price + 1), "1", "1"],
                "b": [str(price - 1), "1", "1"],
                "c": [str(price), "1"],
                "v": ["1000", "1000"],
                "p": [str(price), str(price)],
                "o": "100",
            }
        return {"result": result}

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch_pct
    )

    pairs = [f"PAIR{i}" for i in range(1, 11)]

    class DummyEx:
        markets_by_id = {p: {"symbol": p} for p in pairs}

    symbols = asyncio.run(filter_symbols(DummyEx(), pairs, CONFIG))
    assert {s for s, _ in symbols} == {"PAIR9", "PAIR10"}


def test_get_symbol_age(monkeypatch):
    class AgeExchange:
        def __init__(self):
            self.markets = {"BTC/USD": {"created": 0}}

        def milliseconds(self):
            return 10 * 86400000

    age = sc.get_symbol_age(AgeExchange(), "BTC/USD")
    assert age == 0.0


def test_get_latency(monkeypatch):
    class LatencyExchange:
        def fetch_ticker(self, symbol):
            return {}

    calls = [1.0, 1.2]

    def fake_counter():
        return calls.pop(0)

    monkeypatch.setattr(sc.time, "perf_counter", fake_counter)
    latency = asyncio.run(sc.get_latency(LatencyExchange(), "BTC/USD"))
    assert latency == pytest.approx(200.0)
