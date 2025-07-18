import asyncio
import json
import pandas as pd
import pytest
import ccxt
import crypto_bot.utils.symbol_scoring as sc
import crypto_bot.utils.symbol_pre_filter as sp
from crypto_bot.utils.telemetry import telemetry


@pytest.fixture(autouse=True)
def reset_telemetry():
    telemetry.reset()
    sp.liq_cache.clear()
    yield


@pytest.fixture(autouse=True)
def reset_semaphore():
    sp.SEMA = asyncio.Semaphore(1)
    yield


from crypto_bot.utils.symbol_pre_filter import filter_symbols, has_enough_history

CONFIG = {
    "symbol_filter": {
        "volume_percentile": 0,
        "change_pct_percentile": 0,
        "max_spread_pct": 3.0,
        "correlation_max_pairs": 100,
    },
    "symbol_score_weights": {
        "volume": 1,
        "change": 0,
        "spread": 0,
        "age": 0,
        "latency": 0,
    },
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
    assert symbols == [("ETH/USD", 0.8), ("BTC/USD", 0.6)]


class FetchTickersExchange(DummyExchange):
    def __init__(self):
        self.has = {"fetchTickers": True}
        self.markets_by_id = DummyExchange.markets_by_id

    async def fetch_tickers(self, symbols):
        assert symbols == ["ETH/USD", "BTC/USD"]
        data = (await fake_fetch(None))["result"]
        return {"ETH/USD": data["XETHZUSD"], "BTC/USD": data["XXBTZUSD"]}


def test_filter_symbols_fetch_tickers(monkeypatch):
    async def raise_if_called(_):
        raise AssertionError("_fetch_ticker_async should not be called")

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", raise_if_called
    )

    ex = FetchTickersExchange()

    symbols = asyncio.run(filter_symbols(ex, ["ETH/USD", "BTC/USD"], CONFIG))

    assert symbols == [("ETH/USD", 0.8), ("BTC/USD", 0.6)]


class NormalizedFetchTickersExchange(DummyExchange):
    def __init__(self):
        self.has = {"fetchTickers": True}
        self.markets_by_id = DummyExchange.markets_by_id

    async def fetch_tickers(self, symbols):
        assert symbols == ["ETH/USD", "BTC/USD"]
        data = (await fake_fetch(None))["result"]

        def to_normalized(t):
            return {
                "info": t,
                "bid": float(t["b"][0]),
                "ask": float(t["a"][0]),
                "last": float(t["c"][0]),
                "open": float(t["o"]),
                "vwap": float(t["p"][1]),
                "baseVolume": float(t["v"][1]),
            }

        return {
            "ETH/USD": to_normalized(data["XETHZUSD"]),
            "BTC/USD": to_normalized(data["XXBTZUSD"]),
        }


def test_filter_symbols_fetch_tickers_normalized(monkeypatch):
    async def raise_if_called(_):
        raise AssertionError("_fetch_ticker_async should not be called")

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", raise_if_called
    )

    ex = NormalizedFetchTickersExchange()

    symbols = asyncio.run(filter_symbols(ex, ["ETH/USD", "BTC/USD"], CONFIG))

    assert symbols == [("ETH/USD", 0.8), ("BTC/USD", 0.6)]


class AltnameMappingExchange:
    def __init__(self):
        self.has = {"fetchTickers": True}
        self.markets_by_id = {"XXBTZUSD": {"symbol": "XBT/USDT", "altname": "XBTUSDT"}}

    async def fetch_tickers(self, symbols):
        assert symbols == ["XBT/USDT"]
        data = (await fake_fetch(None))["result"]
        return {"XBT/USDT": data["XXBTZUSD"]}


def test_altname_mapping(monkeypatch):
    async def raise_if_called(_):
        raise AssertionError("_fetch_ticker_async should not be called")

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", raise_if_called
    )

    ex = AltnameMappingExchange()
    symbols = asyncio.run(filter_symbols(ex, ["XBT/USDT"], CONFIG))

    assert symbols == [("BTC/USDT", 0.6)]


class RawIdMappingExchange:
    def __init__(self):
        self.has = {"fetchTickers": True}
        self.markets_by_id = {"XXBTZUSD": {"symbol": "XBT/USDT", "altname": "XBTUSDT"}}

    async def fetch_tickers(self, symbols):
        assert symbols == ["XBTUSDT"]
        data = (await fake_fetch(None))["result"]
        return {"XBTUSDT": data["XXBTZUSD"]}


def test_raw_id_mapping(monkeypatch):
    async def raise_if_called(_):
        raise AssertionError("_fetch_ticker_async should not be called")

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", raise_if_called
    )

    ex = RawIdMappingExchange()
    symbols = asyncio.run(filter_symbols(ex, ["XBTUSDT"], CONFIG))

    assert symbols == [("BTC/USDT", 0.6)]


class USDCVolumeExchange:
    def __init__(self):
        self.has = {"fetchTickers": True}
        self.markets_by_id = {
            "XLOWUSDC": {"symbol": "LOW/USDC"},
            "XLOWUSDT": {"symbol": "LOW/USDT"},
        }

    async def fetch_tickers(self, symbols):
        assert symbols == ["LOW/USDC", "LOW/USDT"]
        ticker = {
            "a": ["101", "1", "1"],
            "b": ["100", "1", "1"],
            "c": ["101", "1"],
            "v": ["300", "300"],
            "p": ["100", "100"],
            "o": "99",
        }
        return {"LOW/USDC": ticker, "LOW/USDT": ticker}


def test_usdc_min_volume_halved(monkeypatch):
    async def raise_if_called(_):
        raise AssertionError("_fetch_ticker_async should not be called")

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", raise_if_called
    )

    ex = USDCVolumeExchange()
    result = asyncio.run(
        filter_symbols(ex, ["LOW/USDC", "LOW/USDT"], CONFIG)
    )

    assert result == [("LOW/USDC", 0.3)]


class WatchTickersExchange(DummyExchange):
    def __init__(self):
        self.has = {"watchTickers": True}
        self.calls = 0
        self.markets_by_id = DummyExchange.markets_by_id

    async def watch_tickers(self, symbols):
        self.calls += 1
        data = (await fake_fetch(None))["result"]
        return {"ETH/USD": data["XETHZUSD"], "BTC/USD": data["XXBTZUSD"]}


def test_watch_tickers_cache(monkeypatch):
    sp.ticker_cache.clear()
    sp.ticker_ts.clear()
    ex = WatchTickersExchange()

    t = {"now": 0}

    monkeypatch.setattr(sp.time, "time", lambda: t["now"])

    symbols = asyncio.run(filter_symbols(ex, ["ETH/USD", "BTC/USD"], CONFIG))
    assert ex.calls == 1
    assert symbols == [("ETH/USD", 0.8), ("BTC/USD", 0.6)]

    symbols = asyncio.run(filter_symbols(ex, ["ETH/USD", "BTC/USD"], CONFIG))
    assert ex.calls == 1

    t["now"] += 6
    symbols = asyncio.run(filter_symbols(ex, ["ETH/USD", "BTC/USD"], CONFIG))
    assert ex.calls == 2


class FailingWatchExchange(DummyExchange):
    def __init__(self):
        self.has = {"watchTickers": True, "fetchTickers": True}
        self.watch_calls = 0
        self.fetch_calls = 0
        self.markets_by_id = DummyExchange.markets_by_id

    async def watch_tickers(self, symbols):
        self.watch_calls += 1
        raise RuntimeError("ws boom")

    async def fetch_tickers(self, symbols):
        self.fetch_calls += 1
        data = (await fake_fetch(None))["result"]
        return {"ETH/USD": data["XETHZUSD"], "BTC/USD": data["XXBTZUSD"]}


def test_watch_tickers_fallback(monkeypatch, caplog, tmp_path):
    caplog.set_level("INFO")
    import crypto_bot.utils.pair_cache as pc

    pair_file = tmp_path / "liquid_pairs.json"
    monkeypatch.setattr(pc, "PAIR_FILE", pair_file)
    monkeypatch.setattr(sp, "PAIR_FILE", pair_file)

    async def raise_if_called(_pairs):
        raise AssertionError("_fetch_ticker_async should not be called")

    monkeypatch.setattr(sp, "_fetch_ticker_async", raise_if_called)

    sp.ticker_cache.clear()
    sp.ticker_ts.clear()
    ex = FailingWatchExchange()

    t = {"now": 0}
    monkeypatch.setattr(sp.time, "time", lambda: t["now"])

    symbols = asyncio.run(filter_symbols(ex, ["ETH/USD", "BTC/USD"], CONFIG))
    assert ex.watch_calls == 1
    assert ex.fetch_calls == 1
    assert symbols == [("ETH/USD", 0.8), ("BTC/USD", 0.6)]
    assert any("falling back" in r.getMessage() for r in caplog.records)
    assert telemetry.snapshot().get("scan.ws_errors", 0) == 1

    symbols = asyncio.run(filter_symbols(ex, ["ETH/USD", "BTC/USD"], CONFIG))
    assert ex.watch_calls == 1
    assert ex.fetch_calls == 1


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


class FailLoadExchange:
    def __init__(self):
        self.markets_by_id = {}

    def load_markets(self):
        raise RuntimeError("boom")


def test_load_markets_failure_fallback(monkeypatch, caplog):
    caplog.set_level("WARNING")
    ex = FailLoadExchange()
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )
    symbols = asyncio.run(filter_symbols(ex, ["ETH/USD"], CONFIG))
    assert symbols == [("ETH/USD", 0.8)]
    assert any("load_markets failed" in r.getMessage() for r in caplog.records)


def test_non_dict_market_entry(monkeypatch):
    class BadExchange:
        markets_by_id = {"XETHZUSD": ["ETH/USD"]}

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )
    symbols = asyncio.run(filter_symbols(BadExchange(), ["ETH/USD"], CONFIG))
    assert symbols == [("ETH/USD", 0.8)]


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
        "symbol_filter": {
            "volume_percentile": 0,
            "change_pct_percentile": 0,
            "max_spread_pct": 2.0,
        },
    }
    symbols = asyncio.run(filter_symbols(DummyExchange(), ["ETH/USD", "BTC/USD"], cfg))

    assert symbols == [("BTC/USD", 0.8), ("ETH/USD", 0.6)]


def test_filter_symbols_min_score(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )

    cfg = {
        **CONFIG,
        "min_symbol_score": 0.7,
        "symbol_filter": {"volume_percentile": 0, "change_pct_percentile": 0},
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


def test_filter_symbols_min_age_uses_cache(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )

    called = False

    async def fake_update(exchange, cache, symbols, **_):
        nonlocal called
        called = True
        return cache

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter.update_ohlcv_cache", fake_update
    )

    cfg = {**CONFIG, "min_symbol_age_days": 2}
    ex = HistoryExchange(10)
    df = pd.DataFrame({"close": range(48)})
    cache = {"ETH/USD": df}
    symbols = asyncio.run(filter_symbols(ex, ["ETH/USD"], cfg, df_cache=cache))
    assert symbols == [("ETH/USD", 0.8)]
    assert not called


def test_filter_symbols_correlation(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async",
        fake_fetch,
    )
    df1 = pd.DataFrame({"close": list(range(1, 32))})
    df1["return"] = df1["close"].pct_change()
    df2 = pd.DataFrame({"close": [2 * i for i in range(1, 32)]})
    df2["return"] = df2["close"].pct_change()
    cache = {"ETH/USD": df1, "BTC/USD": df2}

    cfg = {
        **CONFIG,
        "symbol_filter": {
            "volume_percentile": 0,
            "max_spread_pct": 2.0,
            "change_pct_percentile": 0,
        },
    }
    symbols = asyncio.run(
        filter_symbols(DummyExchange(), ["ETH/USD", "BTC/USD"], cfg, df_cache=cache)
    )

    assert symbols == [("ETH/USD", 0.8)]


def test_correlation_pair_limit(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async",
        fake_fetch,
    )
    df1 = pd.DataFrame({"close": [1, 2, 3]})
    df1["return"] = df1["close"].pct_change()
    df2 = pd.DataFrame({"close": [2, 4, 6]})
    df2["return"] = df2["close"].pct_change()
    cache = {"ETH/USD": df1, "BTC/USD": df2}

    cfg = {
        **CONFIG,
        "symbol_filter": {
            "min_volume_usd": 50000,
            "max_spread_pct": 2.0,
            "change_pct_percentile": 0,
            "volume_percentile": 0,
            "correlation_max_pairs": 1,
        },
    }
    symbols = asyncio.run(
        filter_symbols(DummyExchange(), ["ETH/USD", "BTC/USD"], cfg, df_cache=cache)
    )

    # second symbol is not pruned because correlation checks are limited to 1 pair
    assert symbols == [("ETH/USD", 0.8), ("BTC/USD", 0.6)]


def test_correlation_skipped_when_insufficient_history(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async",
        fake_fetch,
    )
    df1 = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    df1["return"] = df1["close"].pct_change()
    df2 = pd.DataFrame({"close": [2, 4, 6, 8, 10]})
    df2["return"] = df2["close"].pct_change()
    cache = {"ETH/USD": df1, "BTC/USD": df2}

    cfg = {
        **CONFIG,
        "symbol_filter": {
            "volume_percentile": 0,
            "max_spread_pct": 2.0,
            "change_pct_percentile": 0,
        },
    }
    symbols = asyncio.run(
        filter_symbols(DummyExchange(), ["ETH/USD", "BTC/USD"], cfg, df_cache=cache)
    )

    assert symbols == [("ETH/USD", 0.8), ("BTC/USD", 0.6)]


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
    cfg = {"symbol_filter": {"volume_percentile": 0, "max_spread_pct": 1.0}}
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
    assert {s for s, _ in symbols} == set(pairs)


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


def test_symbol_skipped_when_missing_from_cache(monkeypatch, tmp_path):
    pair_file = tmp_path / "liquid_pairs.json"
    pair_file.write_text(json.dumps({"ETH/USD": 0}))
    import crypto_bot.utils.symbol_pre_filter as sp
    import crypto_bot.utils.pair_cache as pc

    monkeypatch.setattr(pc, "PAIR_FILE", pair_file)
    monkeypatch.setattr(sp, "PAIR_FILE", pair_file)

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )

    async def fake_history(*_a, **_k):
        return True

    monkeypatch.setattr(sp, "has_enough_history", fake_history)

    async def fake_update(exchange, cache, symbols, **_):
        for s in symbols:
            cache[s] = pd.DataFrame({"close": range(48)})
        return cache

    monkeypatch.setattr(sp, "update_ohlcv_cache", fake_update)

    cfg = {
        **CONFIG,
        "min_symbol_age_days": 1,
        "symbol_filter": {**CONFIG["symbol_filter"], "uncached_volume_multiplier": 1},
    }
    ex = DummyExchange()
    ex.has = {}
    result = asyncio.run(sp.filter_symbols(ex, ["ETH/USD", "BTC/USD"], cfg))

    assert result == [("ETH/USD", 0.8), ("BTC/USD", 0.6)]


def test_uncached_multiplier_allows_symbol(monkeypatch, tmp_path):
    pair_file = tmp_path / "liquid_pairs.json"
    pair_file.write_text(json.dumps({"ETH/USD": 0}))

    import crypto_bot.utils.symbol_pre_filter as sp
    import crypto_bot.utils.pair_cache as pc

    monkeypatch.setattr(pc, "PAIR_FILE", pair_file)
    monkeypatch.setattr(sp, "PAIR_FILE", pair_file)

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )

    async def fake_history(*_a, **_k):
        return True

    monkeypatch.setattr(sp, "has_enough_history", fake_history)

    async def fake_update(exchange, cache, symbols, **_):
        return {s: pd.DataFrame({"close": [0] * 48}) for s in symbols}

    monkeypatch.setattr(sp, "update_ohlcv_cache", fake_update)

    cfg = {
        **CONFIG,
        "min_symbol_age_days": 1,
        "symbol_filter": {
            **CONFIG["symbol_filter"],
            "uncached_volume_multiplier": 1,
        },
    }
    result = asyncio.run(
        sp.filter_symbols(DummyExchange(), ["ETH/USD", "BTC/USD"], cfg)
    )

    assert result == [("ETH/USD", 0.8), ("BTC/USD", 0.6)]


def test_liq_cache_skips_api(monkeypatch):
    sp.liq_cache.clear()
    sp.liq_cache["ETH/USD"] = (60000.0, 0.5)

    class DummyExchange:
        has = {}
        markets_by_id = {"XETHZUSD": {"symbol": "ETH/USD"}}

        async def fetch_tickers(self, symbols):
            raise AssertionError("fetch_tickers should not be called")

    async def fake_score_symbol(*_a, **_k):
        return 1.0

    monkeypatch.setattr(sp, "score_symbol", fake_score_symbol)
    monkeypatch.setattr(sp, "_fetch_ticker_async", fake_fetch)

    cfg = {
        "symbol_filter": {
            "min_volume_usd": 50000,
            "max_spread_pct": 1.0,
            "change_pct_percentile": 0,
        }
    }
    result = asyncio.run(sp.filter_symbols(DummyExchange(), ["ETH/USD"], cfg))
    assert result == [("ETH/USD", 1.0)]


def test_stale_cache_not_counted_as_skipped(monkeypatch):
    sp.liq_cache.clear()
    sp.liq_cache["ETH/USD"] = (10000.0, 0.5)

    class DummyExchange:
        has = {}
        markets_by_id = {"XETHZUSD": {"symbol": "ETH/USD"}}

    monkeypatch.setattr(sp, "_fetch_ticker_async", fake_fetch)

    result = asyncio.run(sp.filter_symbols(DummyExchange(), ["ETH/USD"], CONFIG))

    assert result == [("ETH/USD", 0.8)]
    assert telemetry.snapshot().get("scan.symbols_skipped", 0) == 0


def test_refresh_tickers_warns_missing_market(monkeypatch, caplog):
    caplog.set_level("WARNING")

    class DummyExchange:
        has = {}
        markets = {}
        called = False

        def load_markets(self):
            self.called = True
            self.markets = {"ETH/USD": {}}

    async def fake_fetch(_pairs):
        return {"result": {}}

    monkeypatch.setattr(sp, "_fetch_ticker_async", fake_fetch)

    ex = DummyExchange()
    asyncio.run(sp._refresh_tickers(ex, ["ETH/USD", "BTC/USD"]))

    assert any("BTC/USD" in r.getMessage() for r in caplog.records)
    assert ex.called is True


def test_refresh_tickers_bad_symbol(monkeypatch, caplog):
    caplog.set_level("WARNING")

    class BadSymbolExchange:
        has = {"fetchTickers": True}
        markets = {"ETH/USD": {}, "BTC/USD": {}}

        async def fetch_tickers(self, symbols):
            raise ccxt.BadSymbol("bad symbol")

    result = asyncio.run(
        sp._refresh_tickers(BadSymbolExchange(), ["ETH/USD", "BTC/USD"])
    )

    assert result == {}
    assert any("BadSymbol" in r.getMessage() for r in caplog.records)
    assert any("BTC/USD" in r.getMessage() for r in caplog.records)


def test_refresh_tickers_retry_520(monkeypatch):
    class RetryExchange(DummyExchange):
        def __init__(self):
            self.has = {"fetchTickers": True}
            self.calls = 0
            self.markets = {"ETH/USD": {}, "BTC/USD": {}}

        async def fetch_tickers(self, symbols):
            self.calls += 1
            if self.calls == 1:
                err = ccxt.ExchangeError("boom")
                err.http_status = 520
                raise err
            data = (await fake_fetch(None))["result"]
            return {"ETH/USD": data["XETHZUSD"], "BTC/USD": data["XXBTZUSD"]}

    sleeps = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    monkeypatch.setattr(sp.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(sp, "_fetch_ticker_async", lambda _p: {"result": {}})

    ex = RetryExchange()
    result = asyncio.run(sp._refresh_tickers(ex, ["ETH/USD", "BTC/USD"]))

    assert ex.calls == 2
    assert sleeps == [1]
    assert set(result) == {"ETH/USD", "BTC/USD"}


def test_refresh_tickers_retry_520_network(monkeypatch):
    class RetryExchange(DummyExchange):
        def __init__(self):
            self.has = {"fetchTickers": True}
            self.calls = 0
            self.markets = {"ETH/USD": {}, "BTC/USD": {}}

        async def fetch_tickers(self, symbols):
            self.calls += 1
            if self.calls == 1:
                err = ccxt.ExchangeNotAvailable("boom")
                err.http_status = 520
                raise err
            data = (await fake_fetch(None))["result"]
            return {"ETH/USD": data["XETHZUSD"], "BTC/USD": data["XXBTZUSD"]}

    sleeps = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    monkeypatch.setattr(sp.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(sp, "_fetch_ticker_async", lambda _p: {"result": {}})

    ex = RetryExchange()
    result = asyncio.run(sp._refresh_tickers(ex, ["ETH/USD", "BTC/USD"]))

    assert ex.calls == 2
    assert sleeps == [1]
    assert set(result) == {"ETH/USD", "BTC/USD"}


def test_refresh_tickers_single_fallback(monkeypatch):
    class FailBothExchange(DummyExchange):
        def __init__(self):
            self.has = {"fetchTickers": True, "fetchTicker": True}
            self.bulk_calls = 0
            self.single_calls: list[str] = []
            self.markets = {"ETH/USD": {}, "BTC/USD": {}}

        async def fetch_tickers(self, symbols):
            self.bulk_calls += 1
            raise RuntimeError("boom")

        async def fetch_ticker(self, symbol):
            self.single_calls.append(symbol)
            data = (await fake_fetch(None))["result"]
            mapping = {"ETH/USD": "XETHZUSD", "BTC/USD": "XXBTZUSD"}
            return data[mapping[symbol]]

    calls: list[list[str]] = []

    async def raise_fetch_async(pairs):
        calls.append(list(pairs))
        raise RuntimeError("boom")

    monkeypatch.setattr(sp, "_fetch_ticker_async", raise_fetch_async)

    ex = FailBothExchange()
    result = asyncio.run(sp._refresh_tickers(ex, ["ETH/USD", "BTC/USD"]))

    assert ex.bulk_calls == 2
    assert calls == [["ETHUSD", "BTCUSD"]]
    assert ex.single_calls == ["ETH/USD", "BTC/USD"]
    assert set(result) == {"ETH/USD", "BTC/USD"}


def test_refresh_tickers_public_api_fallback(monkeypatch):
    class FailingExchange(DummyExchange):
        def __init__(self):
            self.has = {"fetchTickers": True}
            self.bulk_calls = 0
            self.markets = {"ETH/USD": {}, "BTC/USD": {}}

        async def fetch_tickers(self, symbols):
            self.bulk_calls += 1
            raise RuntimeError("boom")

    calls: list[list[str]] = []

    async def fake_public(pairs):
        calls.append(list(pairs))
        return await fake_fetch(None)

    monkeypatch.setattr(sp, "_fetch_ticker_async", fake_public)

    ex = FailingExchange()
    result = asyncio.run(sp._refresh_tickers(ex, ["ETH/USD", "BTC/USD"]))

    assert ex.bulk_calls == 2
    assert calls == [["ETHUSD", "BTCUSD"]]
    assert set(result) == {"ETH/USD", "BTC/USD"}


def test_refresh_tickers_batches(monkeypatch):
    class BatchExchange(DummyExchange):
        def __init__(self):
            self.has = {"fetchTickers": True}
            self.calls: list[list[str]] = []
            self.markets = {f"PAIR{i}/USD": {} for i in range(5)}

        async def fetch_tickers(self, symbols):
            self.calls.append(list(symbols))
            return {s: {} for s in symbols}

    ex = BatchExchange()
    monkeypatch.setattr(sp, "cfg", {"symbol_filter": {"kraken_batch_size": 2, "http_timeout": 10}})

    result = asyncio.run(sp._refresh_tickers(ex, list(ex.markets)))

    assert ex.calls == [["PAIR0/USD", "PAIR1/USD"], ["PAIR2/USD", "PAIR3/USD"], ["PAIR4/USD"]]
    assert set(result) == set(ex.markets)


def test_fetch_ticker_async_timeout(monkeypatch):
    calls: list[int | None] = []

    class FakeResp:
        def raise_for_status(self):
            pass

        async def json(self):
            return {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def get(self, url, timeout=None):
            calls.append(timeout)
            return FakeResp()

    monkeypatch.setattr(sp.aiohttp, "ClientSession", lambda: FakeSession())

    asyncio.run(sp._fetch_ticker_async(["XBTUSD"], timeout=5))

    assert calls == [5]
def test_ticker_retry_attempts(monkeypatch):
    class RetryExchange(DummyExchange):
        def __init__(self):
            self.has = {"fetchTickers": True}
            self.calls = 0
            self.markets = {"ETH/USD": {}, "BTC/USD": {}}

        async def fetch_tickers(self, symbols):
            self.calls += 1
            raise ccxt.ExchangeError("boom")

    sleeps = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    monkeypatch.setattr(sp.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(sp, "_fetch_ticker_async", lambda _p: {"result": {}})

    ex = RetryExchange()
    cfg = {"symbol_filter": {"ticker_retry_attempts": 1}}
    result = asyncio.run(sp._refresh_tickers(ex, ["ETH/USD", "BTC/USD"], cfg))

    assert ex.calls == 2
    assert sleeps == []
    assert result == {}


def test_log_ticker_exceptions(monkeypatch, caplog):
    caplog.set_level("WARNING")

    class FailingExchange(DummyExchange):
        def __init__(self):
            self.has = {"fetchTickers": True}
            self.markets = {"ETH/USD": {}, "BTC/USD": {}}

        async def fetch_tickers(self, symbols):
            raise RuntimeError("boom")

    monkeypatch.setattr(sp, "_fetch_ticker_async", lambda _p: {"result": {}})

    ex = FailingExchange()
    cfg = {"symbol_filter": {"log_ticker_exceptions": True}}
    asyncio.run(sp._refresh_tickers(ex, ["ETH/USD"], cfg))
    assert any(r.exc_info for r in caplog.records)

    caplog.clear()
    cfg["symbol_filter"]["log_ticker_exceptions"] = False
    asyncio.run(sp._refresh_tickers(ex, ["ETH/USD"], cfg))
    assert any(r.exc_info for r in caplog.records)


class AlwaysFailWatchExchange(DummyExchange):
    def __init__(self):
        self.has = {"watchTickers": True, "fetchTickers": True}
        self.watch_calls = 0
        self.fetch_calls = 0
        self.markets_by_id = DummyExchange.markets_by_id
        self.options = {"ws_scan": True}

    async def watch_tickers(self, symbols):
        self.watch_calls += 1
        raise RuntimeError("ws boom")

    async def fetch_tickers(self, symbols):
        self.fetch_calls += 1
        data = (await fake_fetch(None))["result"]
        return {"ETH/USD": data["XETHZUSD"], "BTC/USD": data["XXBTZUSD"]}


def test_ws_failures_disable_scan(monkeypatch):
    monkeypatch.setattr(sp, "_fetch_ticker_async", lambda _p: {"result": {}})

    sp.ticker_cache.clear()
    sp.ticker_ts.clear()
    ex = AlwaysFailWatchExchange()

    t = {"now": 0}
    monkeypatch.setattr(sp.time, "time", lambda: t["now"])

    sleeps: list[float] = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    monkeypatch.setattr(sp.asyncio, "sleep", fake_sleep)

    cfg = dict(CONFIG)
    cfg["ws_failures_before_disable"] = 2

    asyncio.run(sp._refresh_tickers(ex, ["ETH/USD", "BTC/USD"], cfg))
    assert ex.watch_calls == 1
    assert ex.fetch_calls == 1
    assert ex.options.get("ws_failures") == 1
    assert ex.options.get("ws_scan") is True
    assert sleeps == []

    t["now"] += 6
    asyncio.run(sp._refresh_tickers(ex, ["ETH/USD", "BTC/USD"], cfg))
    assert sleeps == [1]
    assert ex.watch_calls == 2
    assert ex.fetch_calls == 1
    assert ex.options.get("ws_failures") == 2
    assert ex.options.get("ws_scan") is False

    t["now"] += 6
    asyncio.run(sp._refresh_tickers(ex, ["ETH/USD", "BTC/USD"], cfg))
    assert ex.watch_calls == 2
    assert ex.fetch_calls == 2
    assert sleeps == [1]
