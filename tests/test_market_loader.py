import asyncio
import pandas as pd
import pytest
import logging
import time
import ccxt

VALID_MINT = "So11111111111111111111111111111111111111112"

from crypto_bot.utils.market_loader import (
    load_kraken_symbols,
    fetch_ohlcv_async,
    fetch_order_book_async,
    load_ohlcv_parallel,
    update_ohlcv_cache,
    update_multi_tf_ohlcv_cache,
    update_regime_tf_cache,
)


class DummyExchange:
    exchange_market_types = {"spot"}

    def load_markets(self):
        return {
            "BTC/USD": {"active": True, "type": "spot"},
            "ETH/USD": {"active": True, "type": "margin"},
            "XBT/USD-PERP": {"active": True, "type": "futures"},
            "XRP/USD": {"active": False, "type": "spot"},
        }


def test_load_kraken_symbols_returns_active():
    ex = DummyExchange()
    symbols = asyncio.run(load_kraken_symbols(ex))
    assert set(symbols) == {"BTC/USD"}


def test_excluded_symbols_are_removed():
    ex = DummyExchange()
    symbols = asyncio.run(load_kraken_symbols(ex, exclude=["ETH/USD"]))
    assert set(symbols) == {"BTC/USD"}


def test_load_kraken_symbols_logs_exclusions(caplog):
    ex = DummyExchange()
    from crypto_bot.utils import market_loader

    with caplog.at_level(logging.DEBUG):
        market_loader.logger.setLevel(logging.DEBUG)
        symbols = asyncio.run(load_kraken_symbols(ex, exclude=["ETH/USD"]))
    assert set(symbols) == {"BTC/USD"}
    messages = [r.getMessage() for r in caplog.records]
    assert any("Skipping symbol XRP/USD" in m for m in messages)
    assert any("Skipping symbol ETH/USD" in m for m in messages)
    assert any("Skipping symbol XBT/USD-PERP" in m for m in messages)
    assert any("Including symbol BTC/USD" in m for m in messages)


def test_load_kraken_symbols_market_type_filter():
    ex = DummyExchange()
    ex.exchange_market_types = {"margin", "futures"}
    symbols = asyncio.run(load_kraken_symbols(ex))
    assert set(symbols) == {"ETH/USD", "XBT/USD-PERP"}


class DummyTypeExchange:
    def load_markets(self):
        return {
            "BTC/USD": {"active": True, "type": "spot"},
            "ETH/USD": {"active": True, "type": "future"},
        }


def test_market_type_filter():
    ex = DummyTypeExchange()
    config = {"exchange_market_types": ["future"]}
    symbols = asyncio.run(load_kraken_symbols(ex, config=config))
    assert symbols == ["ETH/USD"]


class DummySliceExchange:
    has = {"fetchMarketsByType": True}

    def __init__(self):
        self.called: list[str] = []

    def fetch_markets_by_type(self, market_type):
        self.called.append(market_type)
        data = {
            "spot": [
                {"symbol": "BTC/USD", "active": True, "type": "spot"},
            ],
            "future": [
                {"symbol": "XBT/USD-PERP", "active": True, "type": "future"},
            ],
        }
        return data.get(market_type, [])


def test_load_kraken_symbols_fetch_markets_by_type():
    ex = DummySliceExchange()
    ex.exchange_market_types = {"spot", "future"}
    symbols = asyncio.run(load_kraken_symbols(ex))
    assert set(symbols) == {"BTC/USD", "XBT/USD-PERP"}
    assert set(ex.called) == {"spot", "future"}


class DummySymbolFieldExchange:
    exchange_market_types = {"spot"}

    def load_markets(self):
        return {
            "BTC/USD": {"symbol": "BTC/USD", "active": True, "type": "spot"},
            "ETH/USD": {"symbol": "ETH/USD", "active": False, "type": "spot"},
        }


def test_load_kraken_symbols_handles_symbol_column():
    ex = DummySymbolFieldExchange()
    symbols = asyncio.run(load_kraken_symbols(ex))
    assert symbols == ["BTC/USD"]


class DummyAsyncExchange:
    has = {"fetchOHLCV": True}

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[0, 1, 2, 3, 4, 5]]


class DummyWSExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.fetch_called = False

    async def watch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[0] * 6]

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_called = True
        return [[1] * 6 for _ in range(limit)]


class DummyWSExchangeEnough(DummyWSExchange):
    async def watch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[2] * 6 for _ in range(limit)]


def test_fetch_ohlcv_async():
    ex = DummyAsyncExchange()
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD"))
    assert data[0][0] == 0


def test_watch_ohlcv_fallback_to_fetch():
    ex = DummyWSExchange()
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", limit=2, use_websocket=True))
    assert ex.fetch_called is True
    assert len(data) == 2
    assert data[0][0] == 1


def test_watch_ohlcv_no_fallback_when_enough():
    ex = DummyWSExchangeEnough()
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", limit=2, use_websocket=True))
    assert ex.fetch_called is False
    assert len(data) == 2
    assert data[0][0] == 2


class IncompleteExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.calls = 0

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        self.calls += 1
        if self.calls == 1:
            return [[0] * 6 for _ in range(2)]
        return [[1] * 6 for _ in range(limit)]


def test_incomplete_ohlcv_warns_and_retries(caplog):
    ex = IncompleteExchange()
    caplog.set_level(logging.WARNING)
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", limit=5, since=1000))
    assert ex.calls >= 2
    assert len(data) == 5
    assert any(
        "Incomplete OHLCV for BTC/USD: got 2 of 5" in r.getMessage()
        for r in caplog.records
    )


class DummySyncExchange:
    has = {"fetchOHLCV": True}

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[0, 1, 2, 3, 4, 5]]


class DummyBookExchange:
    has = {"fetchOrderBook": True}

    async def fetch_order_book(self, symbol, limit=2):
        return {"bids": [[1, 1], [0.9, 2]], "asks": [[1.1, 1], [1.2, 3]]}


def test_fetch_order_book_async():
    ex = DummyBookExchange()
    data = asyncio.run(fetch_order_book_async(ex, "BTC/USD", depth=2))
    assert data["bids"] == [[1, 1], [0.9, 2]]
    assert data["asks"] == [[1.1, 1], [1.2, 3]]


def test_load_ohlcv_parallel():
    ex = DummySyncExchange()
    result = asyncio.run(
        load_ohlcv_parallel(
            ex,
            ["BTC/USD", "ETH/USD"],
            timeframe="1h",
            limit=1,
            max_concurrent=2,
        )
    )
    assert set(result.keys()) == {"BTC/USD", "ETH/USD"}


class DummyWSEchange:
    has = {"fetchOHLCV": True}

    async def watch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[0, 1, 2, 3, 4, 5]]

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[i, i + 1, i + 2, i + 3, i + 4, i + 5] for i in range(limit)]


class DummyWSExceptionExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.fetch_called = False

    async def watch_ohlcv(self, symbol, timeframe="1h", limit=100):
        raise RuntimeError("ws failed")

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_called = True
        return [[9] * 6 for _ in range(limit)]


def test_load_ohlcv_parallel_websocket_overrides_fetch():
    ex = DummyWSEchange()
    result = asyncio.run(
        load_ohlcv_parallel(
            ex,
            ["BTC/USD"],
            timeframe="1h",
            limit=3,
            use_websocket=True,
            max_concurrent=2,
        )
    )
    assert list(result.keys()) == ["BTC/USD"]
    assert len(result["BTC/USD"]) == 3


def test_load_ohlcv_parallel_websocket_force_history():
    ex = DummyWSEchange()
    result = asyncio.run(
        load_ohlcv_parallel(
            ex,
            ["BTC/USD"],
            timeframe="1h",
            limit=3,
            use_websocket=True,
            force_websocket_history=True,
            max_concurrent=2,
        )
    )
    assert list(result.keys()) == ["BTC/USD"]
    assert len(result["BTC/USD"]) == 1


def test_watch_ohlcv_exception_falls_back_to_fetch():
    ex = DummyWSExceptionExchange()
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", limit=2, use_websocket=True))
    assert ex.fetch_called is True
    assert len(data) == 2
    assert data[0][0] == 9


class LimitCaptureExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.watch_limit = None
        self.fetch_called = False

    async def watch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        self.watch_limit = limit
        return [[0] * 6 for _ in range(limit)]

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        self.fetch_called = True
        return [[1] * 6 for _ in range(limit)]


def test_watch_ohlcv_since_limit_reduction():
    ex = LimitCaptureExchange()
    since = int(time.time() * 1000) - 2 * 3600 * 1000
    data = asyncio.run(
        fetch_ohlcv_async(
            ex,
            "BTC/USD",
            timeframe="1h",
            limit=50,
            since=since,
            use_websocket=True,
        )
    )
    assert ex.fetch_called is False
    assert ex.watch_limit == 1
    assert len(data) == 1


class SkipLargeLimitExchange:
    def __init__(self):
        self.ws_called = False
        self.fetch_called = False

    async def watch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.ws_called = True
        return [[0] * 6 for _ in range(limit)]

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_called = True
        return [[1] * 6 for _ in range(limit)]


def test_watch_ohlcv_skipped_when_limit_exceeds(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = SkipLargeLimitExchange()
    monkeypatch.setattr(market_loader, "MAX_WS_LIMIT", 50)
    data = asyncio.run(
        market_loader.fetch_ohlcv_async(
            ex,
            "BTC/USD",
            timeframe="1m",
            limit=100,
            use_websocket=True,
        )
    )
    assert ex.ws_called is False
    assert ex.fetch_called is True
    assert len(data) == 50


def test_force_websocket_history_ignores_max_ws_limit(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = SkipLargeLimitExchange()
    monkeypatch.setattr(market_loader, "MAX_WS_LIMIT", 50)
    data = asyncio.run(
        market_loader.fetch_ohlcv_async(
            ex,
            "BTC/USD",
            timeframe="1m",
            limit=100,
            use_websocket=True,
            force_websocket_history=True,
        )
    )
    assert ex.ws_called is True
    assert ex.fetch_called is False
    assert len(data) == 100


class DummyIncExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.data = [[i] * 6 for i in range(200)]

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        rows = [r for r in self.data if since is None or r[0] > since]
        return rows[:limit]


class DummyFailExchange(DummyIncExchange):
    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        if since is not None:
            return []
        return await super().fetch_ohlcv(symbol, timeframe, since, limit)


def test_update_ohlcv_cache_appends():
    from crypto_bot.utils import market_loader
    market_loader._last_snapshot_time = 0
    ex = DummyLargeExchange()
    cache: dict[str, pd.DataFrame] = {}
    cache = asyncio.run(
        update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=2, max_concurrent=2)
    )
    assert len(cache["BTC/USD"]) == 200
    ex.data.extend([[i] * 6 for i in range(200, 301)])
    cache = asyncio.run(
        update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=4, max_concurrent=2)
    )
    assert len(cache["BTC/USD"]) == 200


def test_update_ohlcv_cache_fallback_full_history():
    from crypto_bot.utils import market_loader
    market_loader._last_snapshot_time = 0
    ex = DummyFailExchange()
    cache: dict[str, pd.DataFrame] = {}
    cache = asyncio.run(
        update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=3, max_concurrent=2)
    )
    assert len(cache["BTC/USD"]) == 200
    ex.data.extend([[i] * 6 for i in range(200, 301)])
    cache = asyncio.run(
        update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=4, max_concurrent=2)
    )
    assert len(cache["BTC/USD"]) == 200


class DummyLargeExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.data = [[i] * 6 for i in range(200)]

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        rows = [r for r in self.data if since is None or r[0] > since]
        return rows[:limit]


def test_update_ohlcv_cache_enforces_min_limit():
    from crypto_bot.utils import market_loader
    market_loader._last_snapshot_time = 0
    ex = DummyLargeExchange()
    cache: dict[str, pd.DataFrame] = {}
    cache = asyncio.run(
        update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=50, max_concurrent=2)
    )
    assert len(cache["BTC/USD"]) == 200


class CountingExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return [[0] * 6]


def test_load_ohlcv_parallel_respects_max_concurrent():
    ex = CountingExchange()
    symbols = ["A", "B", "C", "D", "E"]
    asyncio.run(
        load_ohlcv_parallel(
            ex,
            symbols,
            limit=1,
            max_concurrent=2,
        )
    )
    assert ex.max_active <= 2


def test_update_ohlcv_cache_respects_max_concurrent():
    ex = CountingExchange()
    symbols = ["A", "B", "C", "D", "E"]
    cache: dict[str, pd.DataFrame] = {}
    asyncio.run(
        update_ohlcv_cache(
            ex,
            cache,
            symbols,
            limit=1,
            max_concurrent=2,
        )
    )
    assert ex.max_active <= 2


def test_load_ohlcv_parallel_invalid_max_concurrent():
    ex = DummySyncExchange()
    with pytest.raises(ValueError):
        asyncio.run(
            load_ohlcv_parallel(
                ex,
                ["BTC/USD"],
                max_concurrent=0,
            )
        )
    with pytest.raises(ValueError):
        asyncio.run(
            load_ohlcv_parallel(
                ex,
                ["BTC/USD"],
                max_concurrent=-1,
            )
        )


class RetryIncompleteExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.fetch_calls = 0

    async def watch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[0] * 6 for _ in range(2)]

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_calls += 1
        if self.fetch_calls == 1:
            return [[1] * 6 for _ in range(4)]
        return [[i] * 6 for i in range(limit)]


class AlwaysIncompleteExchange(RetryIncompleteExchange):
    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_calls += 1
        return [[1] * 6 for _ in range(4)]


def test_update_ohlcv_cache_retry_incomplete_ws():
    ex = RetryIncompleteExchange()
    cache: dict[str, pd.DataFrame] = {}
    res = asyncio.run(
        update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=10,
            use_websocket=True,
            max_concurrent=1,
        )
    )
    assert len(res["BTC/USD"]) == 200
    assert ex.fetch_calls == 2


def test_update_ohlcv_cache_skip_after_retry(caplog):
    ex = AlwaysIncompleteExchange()
    cache: dict[str, pd.DataFrame] = {}
    caplog.set_level(logging.WARNING)
    res = asyncio.run(
        update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=10,
            use_websocket=True,
            max_concurrent=1,
        )
    )
    assert "BTC/USD" not in res
    assert any(
        "Skipping BTC/USD: only 4/200 candles" in r.getMessage() for r in caplog.records
    )


class DummyMultiTFExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.calls: list[str] = []

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.calls.append(timeframe)
        return [[0, 1, 2, 3, 4, 5]]


def test_update_multi_tf_ohlcv_cache():
    ex = DummyMultiTFExchange()
    cache: dict[str, dict[str, pd.DataFrame]] = {}
    config = {"timeframes": ["1h", "4h", "1d"]}
    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            config,
            limit=1,
            max_concurrent=2,
        )
    )
    assert set(cache.keys()) == {"1h", "4h", "1d"}
    for tf in config["timeframes"]:
        assert "BTC/USD" in cache[tf]
    assert set(ex.calls) == {"1h", "4h", "1d"}


def test_update_regime_tf_cache():
    ex = DummyMultiTFExchange()
    cache: dict[str, dict[str, pd.DataFrame]] = {}
    config = {"regime_timeframes": ["5m", "15m", "1h"]}
    df = pd.DataFrame(
        [[0, 1, 2, 3, 4, 5]],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df_map = {"5m": {"BTC/USD": df}, "1h": {"BTC/USD": df}}
    cache = asyncio.run(
        update_regime_tf_cache(
            ex,
            cache,
            ["BTC/USD"],
            config,
            limit=1,
            max_concurrent=2,
            df_map=df_map,
        )
    )
    assert set(cache.keys()) == {"5m", "15m", "1h"}
    for tf in config["regime_timeframes"]:
        assert "BTC/USD" in cache[tf]
    assert set(ex.calls) == {"15m"}


class FailOnceExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.calls = 0

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("fail")
        return [[0] * 6]


class AlwaysFailExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.calls = 0

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.calls += 1
        raise RuntimeError("fail")


class FailSuccessExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.calls = 0

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.calls += 1
        if self.calls % 2 == 1:
            raise RuntimeError("fail")
        return [[0] * 6]


def test_failed_symbol_skipped_until_delay(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = FailOnceExchange()
    cache: dict[str, pd.DataFrame] = {}
    market_loader.failed_symbols.clear()
    monkeypatch.setattr(market_loader, "RETRY_DELAY", 10)

    t = 0

    def fake_time():
        return t

    monkeypatch.setattr(market_loader.time, "time", fake_time)

    asyncio.run(
        market_loader.update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=1,
            max_concurrent=1,
        )
    )
    assert "BTC/USD" not in cache
    assert "BTC/USD" in market_loader.failed_symbols
    assert ex.calls == 1

    asyncio.run(
        market_loader.update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=1,
            max_concurrent=1,
        )
    )
    assert ex.calls == 1  # skipped due to retry delay

    t += 11

    asyncio.run(
        market_loader.update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=1,
            max_concurrent=1,
        )
    )
    assert ex.calls == 2
    assert "BTC/USD" in cache


def test_failed_symbol_backoff(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = AlwaysFailExchange()
    cache: dict[str, pd.DataFrame] = {}
    market_loader.failed_symbols.clear()
    monkeypatch.setattr(market_loader, "RETRY_DELAY", 10)
    monkeypatch.setattr(market_loader, "MAX_RETRY_DELAY", 40)

    t = 0

    def fake_time():
        return t

    monkeypatch.setattr(market_loader.time, "time", fake_time)

    asyncio.run(
        market_loader.update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=1,
            max_concurrent=1,
        )
    )
    assert market_loader.failed_symbols["BTC/USD"]["delay"] == 10

    t += 11
    asyncio.run(
        market_loader.update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=1,
            max_concurrent=1,
        )
    )
    assert market_loader.failed_symbols["BTC/USD"]["delay"] == 20

    t += 21
    asyncio.run(
        market_loader.update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=1,
            max_concurrent=1,
        )
    )
    assert market_loader.failed_symbols["BTC/USD"]["delay"] == 40

    t += 41
    asyncio.run(
        market_loader.update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=1,
            max_concurrent=1,
        )
    )
    assert market_loader.failed_symbols["BTC/USD"]["delay"] == 40


def test_backoff_resets_on_success(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = FailSuccessExchange()
    cache: dict[str, pd.DataFrame] = {}
    market_loader.failed_symbols.clear()
    monkeypatch.setattr(market_loader, "RETRY_DELAY", 10)

    t = 0

    def fake_time():
        return t

    monkeypatch.setattr(market_loader.time, "time", fake_time)

    asyncio.run(
        market_loader.update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=1,
            max_concurrent=1,
        )
    )
    assert market_loader.failed_symbols["BTC/USD"]["delay"] == 10

    t += 11
    asyncio.run(
        market_loader.update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=1,
            max_concurrent=1,
        )
    )
    assert "BTC/USD" in cache
    assert "BTC/USD" not in market_loader.failed_symbols

    t += 1
    ex2 = AlwaysFailExchange()
    asyncio.run(
        market_loader.update_ohlcv_cache(
            ex2,
            cache,
            ["BTC/USD"],
            limit=1,
            max_concurrent=1,
        )
    )
    assert market_loader.failed_symbols["BTC/USD"]["delay"] == 10


def test_symbol_disabled_after_max_failures(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = AlwaysFailExchange()
    market_loader.failed_symbols.clear()
    monkeypatch.setattr(market_loader, "RETRY_DELAY", 0)
    monkeypatch.setattr(market_loader, "MAX_OHLCV_FAILURES", 2)

    asyncio.run(market_loader.load_ohlcv_parallel(ex, ["BTC/USD"], max_concurrent=1))
    assert market_loader.failed_symbols["BTC/USD"]["count"] == 1
    assert not market_loader.failed_symbols["BTC/USD"].get("disabled")

    asyncio.run(market_loader.load_ohlcv_parallel(ex, ["BTC/USD"], max_concurrent=1))
    assert market_loader.failed_symbols["BTC/USD"]["count"] == 2
    assert market_loader.failed_symbols["BTC/USD"].get("disabled") is True

    calls = ex.calls
    asyncio.run(market_loader.load_ohlcv_parallel(ex, ["BTC/USD"], max_concurrent=1))
    assert ex.calls == calls


class StopLoop(Exception):
    pass


def test_main_preserves_symbols_on_scan_failure(monkeypatch, caplog):
    import sys, types

    monkeypatch.setitem(sys.modules, "ccxt", types.SimpleNamespace())
    import crypto_bot.main as main

    caplog.set_level(logging.WARNING)

    async def fake_loader(exchange, exclude=None, config=None):
        main.logger.warning("symbol scan empty")
        return []

    cfg = {"symbol": "BTC/USD", "scan_markets": True}

    monkeypatch.setattr(main, "load_config", lambda: cfg)
    calls = {"loader": 0}

    async def loader_wrapper(*a, **k):
        calls["loader"] += 1
        return await fake_loader(*a, **k)

    monkeypatch.setattr(main, "load_kraken_symbols", loader_wrapper)
    monkeypatch.setattr(main, "cooldown_configure", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "dotenv_values", lambda path: {})
    monkeypatch.setattr(main, "load_or_create", lambda: {})
    monkeypatch.setattr(main, "send_test_message", lambda *_a, **_k: True)
    monkeypatch.setattr(main, "log_balance", lambda *_a, **_k: None)

    class DummyRC:
        def __init__(self, *_a, **_k):
            pass

    class DummyRM:
        def __init__(self, *_a, **_k):
            raise StopLoop

    monkeypatch.setattr(main, "RiskConfig", DummyRC)
    monkeypatch.setattr(main, "RiskManager", DummyRM)
    monkeypatch.setattr(main.asyncio, "sleep", lambda *_a: None)
    monkeypatch.setattr(main, "MAX_SYMBOL_SCAN_ATTEMPTS", 2)
    monkeypatch.setattr(main, "SYMBOL_SCAN_RETRY_DELAY", 0)
    monkeypatch.setattr(main, "MAX_SYMBOL_SCAN_DELAY", 0)

    captured = {}

    class DummyExchange:
        def fetch_balance(self):
            return {"USDT": {"free": 0}}

    def fake_get_exchange(config):
        captured["cfg"] = config
        return DummyExchange(), None

    monkeypatch.setattr(main, "get_exchange", fake_get_exchange)

    asyncio.run(main.main())

    assert "symbols" not in captured["cfg"]
    assert any("aborting startup" in r.getMessage() for r in caplog.records)
    assert calls["loader"] == 2


class SlowExchange:
    has = {"fetchOHLCV": True}

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        await asyncio.sleep(0.05)
        return [[0] * 6]


class SlowWSExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.fetch_called = False

    async def watch_ohlcv(self, symbol, timeframe="1h", limit=100):
        await asyncio.sleep(0.05)
        return [[0] * 6]

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_called = True
        return [[7] * 6 for _ in range(limit)]


def test_fetch_ohlcv_async_timeout(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = SlowExchange()
    monkeypatch.setattr(market_loader, "OHLCV_TIMEOUT", 0.01)
    monkeypatch.setattr(market_loader, "WS_OHLCV_TIMEOUT", 0.01)
    monkeypatch.setattr(market_loader, "REST_OHLCV_TIMEOUT", 0.01)

    res = asyncio.run(market_loader.fetch_ohlcv_async(ex, "BTC/USD"))
    assert isinstance(res, asyncio.TimeoutError)


def test_load_ohlcv_parallel_timeout(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = SlowExchange()
    market_loader.failed_symbols.clear()
    monkeypatch.setattr(market_loader, "OHLCV_TIMEOUT", 0.01)
    monkeypatch.setattr(market_loader, "WS_OHLCV_TIMEOUT", 0.01)
    monkeypatch.setattr(market_loader, "REST_OHLCV_TIMEOUT", 0.01)

    result = asyncio.run(
        market_loader.load_ohlcv_parallel(ex, ["BTC/USD"], max_concurrent=1)
    )
    assert result == {}
    assert "BTC/USD" in market_loader.failed_symbols


def test_fetch_ohlcv_async_timeout_fallback(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = SlowWSExchange()
    monkeypatch.setattr(market_loader, "OHLCV_TIMEOUT", 0.01)
    monkeypatch.setattr(market_loader, "WS_OHLCV_TIMEOUT", 0.01)
    monkeypatch.setattr(market_loader, "REST_OHLCV_TIMEOUT", 0.01)

    data = asyncio.run(
        market_loader.fetch_ohlcv_async(ex, "BTC/USD", limit=2, use_websocket=True)
    )
    assert ex.fetch_called is True
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0][0] == 7


def test_load_ohlcv_parallel_timeout_fallback(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = SlowWSExchange()
    market_loader.failed_symbols.clear()
    monkeypatch.setattr(market_loader, "OHLCV_TIMEOUT", 0.01)
    monkeypatch.setattr(market_loader, "WS_OHLCV_TIMEOUT", 0.01)
    monkeypatch.setattr(market_loader, "REST_OHLCV_TIMEOUT", 0.01)

    result = asyncio.run(
        market_loader.load_ohlcv_parallel(
            ex,
            ["BTC/USD"],
            use_websocket=True,
            limit=2,
            max_concurrent=1,
        )
    )
    assert "BTC/USD" in result
    assert ex.fetch_called is True
    assert "BTC/USD" not in market_loader.failed_symbols


class LimitCaptureWS:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.limit = None

    async def watch_ohlcv(self, symbol, timeframe="1h", limit=100, since=None):
        self.limit = limit
        return [[0] * 6]


def test_watch_ohlcv_since_reduces_limit(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = LimitCaptureWS()
    monkeypatch.setattr(market_loader.time, "time", lambda: 1000)
    since = 1000 - 3 * 3600
    asyncio.run(
        market_loader.fetch_ohlcv_async(
            ex,
            "BTC/USD",
            timeframe="1h",
            limit=10,
            since=since,
            use_websocket=True,
        )
    )
    assert ex.limit == 4


class RateLimitExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.times: list[float] = []
        self.rateLimit = 50

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.times.append(time.time())
        return [[0] * 6]


def test_load_ohlcv_parallel_rate_limit_sleep():
    ex = RateLimitExchange()
    asyncio.run(
        load_ohlcv_parallel(
            ex,
            ["A", "B"],
            limit=1,
            max_concurrent=1,
        )
    )
    assert len(ex.times) == 2
    assert ex.times[1] - ex.times[0] >= ex.rateLimit / 1000


class SymbolCheckExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.symbols: list[str] = []
        self.calls: list[str] = []
        self.loaded = False

    def load_markets(self):
        self.loaded = True
        self.symbols = ["BTC/USD"]

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.calls.append(symbol)
        return [[0] * 6]


def test_invalid_symbol_skipped(caplog):
    from crypto_bot.utils import market_loader

    ex = SymbolCheckExchange()
    caplog.set_level(logging.WARNING)
    result = asyncio.run(
        market_loader.load_ohlcv_parallel(
            ex,
            ["BTC/USD", "ETH/USD"],
            max_concurrent=1,
        )
    )
    assert ex.loaded is True
    assert ex.calls == ["BTC/USD"]
    assert result == {"BTC/USD": [[0] * 6]}
    assert any(
        "Skipping unsupported symbol ETH/USD" in r.getMessage() for r in caplog.records
    )


def test_invalid_symbol_marked_disabled():
    from crypto_bot.utils import market_loader

    ex = SymbolCheckExchange()
    market_loader.failed_symbols.clear()
    result = asyncio.run(market_loader.fetch_ohlcv_async(ex, "ETH/USD"))
    assert result is market_loader.UNSUPPORTED_SYMBOL
    assert market_loader.failed_symbols["ETH/USD"].get("disabled") is True


class MissingTFExchange:
    has = {"fetchOHLCV": True}
    timeframes = {"5m": "5m"}

    def __init__(self):
        self.called = False

    async def fetch_ohlcv(self, *args, **kwargs):
        self.called = True
        return [[0] * 6]


def test_fetch_ohlcv_async_skips_unsupported_timeframe():
    ex = MissingTFExchange()
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", timeframe="1m"))
    assert data == []
    assert ex.called is False


def test_fetch_ohlcv_retry_520(monkeypatch):
    from crypto_bot.utils import market_loader

    class RetryExchange:
        has = {"fetchOHLCV": True}

        def __init__(self):
            self.calls = 0

        async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
            self.calls += 1
            if self.calls == 1:
                err = ccxt.ExchangeError("boom")
                err.http_status = 520
                raise err
            return [[1] * 6]

    sleeps: list[float] = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    ex = RetryExchange()
    monkeypatch.setattr(market_loader.asyncio, "sleep", fake_sleep)
    data = asyncio.run(market_loader.fetch_ohlcv_async(ex, "BTC/USD"))

    assert ex.calls == 2
    assert sleeps == [1]
    assert data == [[1] * 6]


def test_fetch_ohlcv_retry_520_network(monkeypatch):
    from crypto_bot.utils import market_loader

    class RetryExchange:
        has = {"fetchOHLCV": True}

        def __init__(self):
            self.calls = 0

        async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
            self.calls += 1
            if self.calls == 1:
                err = ccxt.NetworkError("boom")
                err.http_status = 520
                raise err
            return [[1] * 6]

    sleeps: list[float] = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    ex = RetryExchange()
    monkeypatch.setattr(market_loader.asyncio, "sleep", fake_sleep)
    data = asyncio.run(market_loader.fetch_ohlcv_async(ex, "BTC/USD"))

    assert ex.calls == 2
    assert sleeps == [1]
    assert data == [[1] * 6]


class CancelWSExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.fetch_called = False
        self.closed = False

    async def watch_ohlcv(self, symbol, timeframe="1h", limit=100):
        raise asyncio.CancelledError

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_called = True
        return [[0] * 6]

    async def close(self):
        self.closed = True


class CancelWSSyncCloseExchange(CancelWSExchange):
    def close(self):
        self.closed = True


def test_fetch_ohlcv_async_cancelled_error():
    ex = CancelWSExchange()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", use_websocket=True, limit=1))
    assert ex.fetch_called is False
    assert ex.closed is True


def test_fetch_ohlcv_async_cancelled_error_sync_close():
    ex = CancelWSSyncCloseExchange()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", use_websocket=True, limit=1))
    assert ex.fetch_called is False
    assert ex.closed is True


def test_load_ohlcv_parallel_cancelled_error(monkeypatch, caplog):
    from crypto_bot.utils import market_loader

    ex = CancelWSExchange()
    market_loader.failed_symbols.clear()
    caplog.set_level(logging.ERROR)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            market_loader.load_ohlcv_parallel(
                ex,
                ["BTC/USD"],
                use_websocket=True,
                limit=1,
                max_concurrent=1,
            )
        )
    assert "BTC/USD" not in market_loader.failed_symbols
    assert not caplog.records
    assert ex.closed is True


class CancelExchange:
    has = {"fetchOHLCV": True}

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        raise asyncio.CancelledError()


def test_load_ohlcv_parallel_propagates_cancelled(caplog):
    ex = CancelExchange()
    caplog.set_level(logging.ERROR)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(load_ohlcv_parallel(ex, ["BTC/USD"], max_concurrent=1))
    assert len(caplog.records) == 0


class PendingWSExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.closed = False
        self.calls: list[str] = []

    async def watch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.calls.append("watch")
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.calls.append("cancel")
            raise
        return [[0] * 6]

    async def close(self):
        self.closed = True
        self.calls.append("close")


def test_fetch_ohlcv_async_cancel_pending_ws(caplog):
    from crypto_bot.utils import market_loader

    async def runner():
        ex = PendingWSExchange()
        task = asyncio.create_task(
            market_loader.fetch_ohlcv_async(
                ex, "BTC/USD", use_websocket=True, limit=1
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return ex.calls

    caplog.set_level(logging.ERROR)
    calls = asyncio.run(runner())
    assert "watch" in calls
    assert not caplog.records


def test_fetch_geckoterminal_ohlcv_success(monkeypatch):
    from crypto_bot.utils import market_loader

    mint = "So11111111111111111111111111111111111111112"

    pool_data = {
        "data": [
            {
                "id": "pool1",
                "attributes": {"volume_usd": {"h24": 123}, "reserve_in_usd": 0},
            },
            {
                "id": "pool1",
                "attributes": {"volume_usd": {"h24": 123}, "address": "pool1"},
            },
            {"id": "solana_pool1", "attributes": {"volume_usd": {"h24": 123}}},
        ]
    }
    ohlcv_data = {
        "data": {
            "attributes": {
                "ohlcv_list": [
                    [1, 1, 2, 0.5, 1.5, 10],
                    [1, 1, 2, 0.5, 1.5, 10],
                ]
            }
        }
    }

    class PoolResp:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def raise_for_status(self):
            pass

        async def json(self):
            return pool_data

    class OhlcvResp(PoolResp):
        async def json(self):
            return ohlcv_data

    class FakeSession:
        calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, timeout=None):
            FakeSession.calls += 1
            if FakeSession.calls == 1:
                assert "search/pools" in url
                assert f"query={mint}%2FUSDC" in url
                return PoolResp()
            assert "/pools/pool1/ohlcv/1h" in url
            return OhlcvResp()

    monkeypatch.setattr(market_loader.aiohttp, "ClientSession", lambda: FakeSession())

    data, vol, reserve = asyncio.run(
        market_loader.fetch_geckoterminal_ohlcv(f"{mint}/USDC", timeframe="1h", limit=1)
    )
    assert data == [[1, 1.0, 2.0, 0.5, 1.5, 10.0]]
    assert vol == 123.0
    assert reserve == 0.0
    data, volume, reserve = asyncio.run(
        market_loader.fetch_geckoterminal_ohlcv(f"{mint}/USDC", timeframe="1h", limit=1)
    )
    assert data == [[1000, 1.0, 2.0, 0.5, 1.5, 10.0]]
    assert volume == 123
    assert reserve == 0


def test_fetch_geckoterminal_ohlcv_invalid_mint(monkeypatch):
    from crypto_bot.utils import market_loader

    def fail_session(*_a, **_k):
        raise AssertionError("should not be called")

    monkeypatch.setattr(market_loader.aiohttp, "ClientSession", fail_session)

    res = asyncio.run(market_loader.fetch_geckoterminal_ohlcv("FOO/USDC"))
    assert res is None


def test_fetch_geckoterminal_ohlcv_invalid_length(monkeypatch):
    from crypto_bot.utils import market_loader

    def fail_session(*_a, **_k):
        raise AssertionError("should not be called")

    monkeypatch.setattr(market_loader.aiohttp, "ClientSession", fail_session)

    res = asyncio.run(market_loader.fetch_geckoterminal_ohlcv("abcd/USDC"))
    assert res is None


def test_fetch_geckoterminal_ohlcv_404(monkeypatch, caplog):
    from crypto_bot.utils import market_loader

    mint = "So11111111111111111111111111111111111111112"

    class FakeResp:
        status = 404

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def raise_for_status(self):
            raise AssertionError("should not be called")

        async def json(self):
            return {}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, timeout=None):
            assert "search/pools" in url
            assert f"query={mint}%2FUSDC" in url
            return FakeResp()

    monkeypatch.setattr(market_loader.aiohttp, "ClientSession", lambda: FakeSession())

    caplog.set_level(logging.INFO)
    res = asyncio.run(market_loader.fetch_geckoterminal_ohlcv(f"{mint}/USDC"))
    assert res is None
    assert any(
        "token not available on GeckoTerminal" in r.getMessage() for r in caplog.records
    )
    assert any(
        "pair not available on GeckoTerminal" in r.getMessage() for r in caplog.records
    )


def test_fetch_geckoterminal_ohlcv_network_error(monkeypatch):
    from crypto_bot.utils import market_loader

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, *a, **k):
            raise market_loader.aiohttp.ClientError("boom")

    monkeypatch.setattr(market_loader.aiohttp, "ClientSession", lambda: FakeSession())

    res = asyncio.run(market_loader.fetch_geckoterminal_ohlcv("FOO/USDC"))
    assert res is None


def test_fetch_geckoterminal_ohlcv_retry(monkeypatch):
    from crypto_bot.utils import market_loader

    sleeps: list[float] = []

    async def fake_sleep(secs):
        sleeps.append(secs)

    class FakeResp:
        status = 200

        def __init__(self, url):
            self.url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def raise_for_status(self):
            pass

        async def json(self):
            if "search/pools" in self.url:
                return {
                    "data": [
                        {
                            "id": "pool1",
                            "attributes": {
                                "volume_usd": {"h24": 123},
                                "reserve_in_usd": 0,
                            },
                        }
                    ]
                }
            return {"data": {"attributes": {"ohlcv_list": [[1, 1, 2, 0.5, 1.5, 10]]}}}

    class RetrySession:
        calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, timeout=None):
            RetrySession.calls += 1
            if RetrySession.calls < 3:
                raise market_loader.aiohttp.ClientError("boom")
            return FakeResp(url)

    monkeypatch.setattr(market_loader.aiohttp, "ClientSession", lambda: RetrySession())
    monkeypatch.setattr(market_loader.asyncio, "sleep", fake_sleep)

    data, vol, reserve = asyncio.run(
        market_loader.fetch_geckoterminal_ohlcv(f"{VALID_MINT}/USDC", limit=1)
    )

    assert RetrySession.calls >= 3
    assert sleeps == [1, 2]
    assert data == [[1, 1.0, 2.0, 0.5, 1.5, 10.0]]
    assert vol == 123.0


def test_geckoterminal_semaphore_limits(monkeypatch):
    from crypto_bot.utils import market_loader

    market_loader.configure(gecko_limit=1)
    market_loader.GECKO_POOL_CACHE.clear()

    calls = {"active": 0, "max": 0}

    class FakeResp:
        status = 200

        def __init__(self, url):
            self.url = url

        async def __aenter__(self):
            calls["max"] = max(calls["max"], calls["active"])
            return self

        async def __aexit__(self, exc_type, exc, tb):
            calls["active"] -= 1

        def raise_for_status(self):
            pass

        async def json(self):
            if "search/pools" in self.url:
                return {
                    "data": [{"id": "pool1", "attributes": {"volume_usd": {"h24": 0}}}]
                }
            return {"data": {"attributes": {"ohlcv_list": [[1, 1, 1, 1, 1, 1]]}}}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        def get(self, url, timeout=None):
            calls["active"] += 1
            return FakeResp(url)

    monkeypatch.setattr(market_loader.aiohttp, "ClientSession", lambda: FakeSession())

    async def worker():
        await market_loader.fetch_geckoterminal_ohlcv(f"{VALID_MINT}/USDC", limit=1)

    async def main():
        await asyncio.gather(*(worker() for _ in range(3)))

    asyncio.run(main())

    assert calls["max"] == 1


def test_update_multi_tf_ohlcv_cache_skips_404(monkeypatch):
    from crypto_bot.utils import market_loader

    async def fake_fetch(*_a, **_k):
        return None

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_fetch)
    monkeypatch.setattr(market_loader, "fetch_coingecko_ohlc", lambda *a, **k: None)

    async def fake_ohlcv(*a, **k):
        return [[1, 1, 1, 1, 1, 1]]

    monkeypatch.setattr(market_loader, "fetch_ohlcv_async", fake_ohlcv)

    ex = DummyMultiTFExchange()
    cache = {}
    config = {"timeframes": ["1h"]}
    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            cache,
            ["FOO/USDC"],
            config,
            limit=1,
        )
    )
    assert "FOO/USDC" in cache["1h"]


def test_update_multi_tf_ohlcv_cache_min_volume(monkeypatch):
    from crypto_bot.utils import market_loader

    async def fake_fetch(*_a, **_k):
        return [[0, 1, 2, 3, 4, 5]], 50.0, 1000.0

    calls: list[float] = []

    async def fake_fetch(*_a, min_24h_volume=0, **_k):
        calls.append(min_24h_volume)
        if min_24h_volume > 50:
            return None
        return [[0, 1, 2, 3, 4, 5]]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_fetch)
    monkeypatch.setattr(market_loader, "fetch_coingecko_ohlc", lambda *a, **k: None)

    async def fake_ohlcv2(*a, **k):
        return [[1, 1, 1, 1, 1, 1]]

    monkeypatch.setattr(market_loader, "fetch_ohlcv_async", fake_ohlcv2)

    ex = DummyMultiTFExchange()
    cache = {}
    config = {"timeframes": ["1h"]}

    config["min_volume_usd"] = 100
    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            cache,
            [f"{VALID_MINT}/USDC"],
            config,
            limit=1,
        )
    )
    assert calls[-1] == 100
    assert f"{VALID_MINT}/USDC" in cache["1h"]
    assert f"{VALID_MINT}/USDC" in cache.get("1h", {})

    config["min_volume_usd"] = 10
    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            cache,
            [f"{VALID_MINT}/USDC"],
            config,
            limit=1,
        )
    )
    assert calls[-1] == 10
    assert f"{VALID_MINT}/USDC" in cache["1h"]


def test_dex_fetch_fallback_coingecko(monkeypatch):
    from crypto_bot.utils import market_loader

    async def fail_gecko(*_a, **_k):
        raise Exception("boom")

    async def fake_coingecko(*_a, **_k):
        return [[0, 1, 2, 3, 4, 5]]

    calls = {"kraken": 0}

    async def fake_fetch(ex, *a, **k):
        calls["kraken"] += 1
        return [[9, 9, 9, 9, 9, 9]]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fail_gecko)
    monkeypatch.setattr(market_loader, "fetch_coingecko_ohlc", fake_coingecko)
    monkeypatch.setattr(market_loader, "fetch_ohlcv_async", fake_fetch)

    ex = DummyMultiTFExchange()
    cache = {}
    config = {"timeframes": ["1h"]}

    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            cache,
            [f"{VALID_MINT}/USDC"],
            config,
            limit=1,
        )
    )
    assert f"{VALID_MINT}/USDC" in cache["1h"]
    assert calls["kraken"] == 0


def test_dex_fetch_fallback_coinbase(monkeypatch):
    from crypto_bot.utils import market_loader

    async def fail_gecko(*_a, **_k):
        return None

    async def fail_coingecko(*_a, **_k):
        return None

    calls = {"coinbase": 0, "exchange": 0}

    async def fake_fetch(ex, *a, **k):
        if getattr(ex, "id", None) == "coinbase":
            calls["coinbase"] += 1
        else:
            calls["exchange"] += 1
        return [[8, 8, 8, 8, 8, 8]]

    class DummyCB:
        id = "coinbase"
        has = {"fetchOHLCV": True}

    monkeypatch.setattr(market_loader.ccxt, "coinbase", lambda params=None: DummyCB())
    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fail_gecko)
    monkeypatch.setattr(market_loader, "fetch_coingecko_ohlc", fail_coingecko)
    monkeypatch.setattr(market_loader, "fetch_ohlcv_async", fake_fetch)

    ex = DummyMultiTFExchange()

    data = asyncio.run(market_loader.fetch_dex_ohlcv(ex, "FOO/USDC", limit=1))
    assert data
    assert calls["coinbase"] == 1
    assert calls["exchange"] == 0


def test_dex_fetch_fallback_kraken(monkeypatch):
    from crypto_bot.utils import market_loader

    async def fail_gecko(*_a, **_k):
        return None

    async def fail_coingecko(*_a, **_k):
        return None

    calls = {"coinbase": 0, "exchange": 0}

    async def fake_fetch(ex, *a, **k):
        if getattr(ex, "id", None) == "coinbase":
            calls["coinbase"] += 1
        else:
            calls["exchange"] += 1
        return [[8, 8, 8, 8, 8, 8]]

    class DummyCB:
        id = "coinbase"
        has = {"fetchOHLCV": True}

    monkeypatch.setattr(market_loader.ccxt, "coinbase", lambda params=None: DummyCB())
    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fail_gecko)
    monkeypatch.setattr(market_loader, "fetch_coingecko_ohlc", fail_coingecko)
    monkeypatch.setattr(market_loader, "fetch_ohlcv_async", fake_fetch)

    ex = DummyMultiTFExchange()

    data = asyncio.run(market_loader.fetch_dex_ohlcv(ex, "FOO/BTC", limit=1))
    assert data
    assert calls["coinbase"] == 0
    assert calls["exchange"] == 1


def test_update_multi_tf_ohlcv_cache_fallback_exchange(monkeypatch):
    from crypto_bot.utils import market_loader

    async def fail_gecko(*_a, **_k):
        raise Exception("boom")

    calls = {"fetch": 0}

    async def fake_fetch(ex, *a, **k):
        calls["fetch"] += 1
        return [[7, 7, 7, 7, 7, 7]]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fail_gecko)
    monkeypatch.setattr(market_loader, "fetch_dex_ohlcv", lambda *a, **k: None)
    monkeypatch.setattr(market_loader, "fetch_ohlcv_async", fake_fetch)

    ex = DummyMultiTFExchange()
    cache = {}
    config = {"timeframes": ["1h"]}


def test_gecko_volume_priority(monkeypatch):
    from crypto_bot.utils import market_loader
    from collections import deque

    async def fake_gecko(*_a, **_k):
        return (
            [
                [0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [2, 1, 1, 1, 1, 10],
            ],
            100.0,
            0.0,
        )

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_gecko)
    ex = DummyMultiTFExchange()
    cache = {}
    q = deque()
    config = {
        "timeframes": ["1h"],
        "bounce_scalper": {"vol_zscore_threshold": 1.0},
    }

    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            cache,
            ["BAR/USDC"],
            config,
            limit=1,
        )
    )
    assert "BAR/USDC" in cache["1h"]
    assert calls["fetch"] == 1
    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            cache,
            ["FOO/USDC"],
            config,
            limit=3,
            priority_queue=q,
        )
    )
    assert "FOO/USDC" in cache["1h"]
    assert list(q) == ["FOO/USDC"]
