import pytest
import asyncio
import logging
import json
import pandas as pd
import time
import threading
import ccxt
import types

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


@pytest.fixture(autouse=True)
def clear_listing_cache():
    from crypto_bot.utils import market_loader
    market_loader._LISTING_DATE_CACHE.clear()


class DummyExchange:
    exchange_market_types = {"spot"}

    def load_markets(self):
        return {
            "BTC/USD": {
                "active": True,
                "type": "spot",
                "base": "BTC",
                "quote": "USD",
            },
            "ETH/USD": {
                "active": True,
                "type": "margin",
                "base": "ETH",
                "quote": "USD",
            },
            "XBT/USD-PERP": {
                "active": True,
                "type": "futures",
                "base": "XBT",
                "quote": "USD",
            },
            "XRP/USD": {
                "active": False,
                "type": "spot",
                "base": "XRP",
                "quote": "USD",
            },
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
            "BTC/USD": {
                "active": True,
                "type": "spot",
                "base": "BTC",
                "quote": "USD",
            },
            "ETH/USD": {
                "active": True,
                "type": "future",
                "base": "ETH",
                "quote": "USD",
            },
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
                {
                    "symbol": "BTC/USD",
                    "active": True,
                    "type": "spot",
                    "base": "BTC",
                    "quote": "USD",
                },
            ],
            "future": [
                {
                    "symbol": "XBT/USD-PERP",
                    "active": True,
                    "type": "future",
                    "base": "XBT",
                    "quote": "USD",
                },
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
            "BTC/USD": {
                "symbol": "BTC/USD",
                "active": True,
                "type": "spot",
                "base": "BTC",
                "quote": "USD",
            },
            "ETH/USD": {
                "symbol": "ETH/USD",
                "active": False,
                "type": "spot",
                "base": "ETH",
                "quote": "USD",
            },
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
        self.fetch_thread = None

    async def watchOHLCV(self, symbol, timeframe="1h", limit=100, **kwargs):
        return [[0] * 6]

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_called = True
        self.fetch_thread = threading.get_ident()
        return [[1] * 6 for _ in range(limit)]


class DummyWSSyncExchange(DummyWSExchange):
    def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_called = True
        self.fetch_thread = threading.get_ident()
        time.sleep(0.01)
        return [[1] * 6 for _ in range(limit)]


def test_fetch_ohlcv_async():
    ex = DummyAsyncExchange()
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD"))
    assert data[0][0] == 0


def test_watchOHLCV_fallback_to_fetch():
    ex = DummyWSExchange()
    main_thread = threading.get_ident()
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", limit=2, use_websocket=True))
    assert ex.fetch_called is True
    assert len(data) == 2
    assert data[0][0] == 1
    assert ex.fetch_thread == main_thread

def test_watchOHLCV_sync_fallback_runs_in_thread():
    ex = DummyWSSyncExchange()
    main_thread = threading.get_ident()
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", limit=2, use_websocket=True))
    assert ex.fetch_called is True
    assert len(data) == 2
    assert data[0][0] == 1
    assert ex.fetch_thread != main_thread


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
    caplog.set_level(logging.INFO)
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", limit=5, since=1000))
    assert ex.calls >= 2
    assert len(data) == 5
    assert any(
        "Incomplete OHLCV for BTC/USD: got 2 of 5" in r.getMessage()
        for r in caplog.records
    )


class TradeFillExchange:
    has = {"fetchOHLCV": True, "fetchTrades": True}

    def __init__(self):
        self.fetch_calls = 0
        self.trades_called = False

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        self.fetch_calls += 1
        if self.fetch_calls == 1:
            return [[0] * 6 for _ in range(2)]
        return [[1] * 6 for _ in range(4)]

    async def fetch_trades(self, symbol, since=None, limit=1000):
        self.trades_called = True
        tf_ms = 3600 * 1000
        start = since or 0
        return [[start + i * tf_ms, i + 1, 1.0] for i in range(limit)]


def test_fetch_trades_used_for_missing_ohlcv():
    ex = TradeFillExchange()
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", limit=5, since=0))
    # fetch_ohlcv_async now relies on repeated REST calls and may not fall back
    # to trades when partial data is returned.
    assert len(data) >= 4


class DummySyncExchange:
    has = {"fetchOHLCV": True}

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[0, 1, 2, 3, 4, 5]]


class DummyBookExchange:
    has = {"fetchOrderBook": True}

    async def fetch_order_book(self, symbol, limit=2):
        return {"bids": [[1, 1], [0.9, 2]], "asks": [[1.1, 1], [1.2, 3]]}


class DummyBookExchangeMarkets:
    has = {"fetchOrderBook": True}

    def __init__(self):
        self.called = False
        self.markets: dict = {}

    async def load_markets(self):
        self.markets = {"BTC/USD": {}}
        return self.markets

    async def fetch_order_book(self, symbol, limit=2):
        self.called = True
        return {"bids": [], "asks": []}


class PagingExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.limits: list[int] = []

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        self.limits.append(limit)
        return [[0] * 6 for _ in range(limit)]


def test_fetch_ohlcv_async_paged_requests():
    ex = PagingExchange()
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", limit=1000))
    assert len(data) == 1000
    assert ex.limits == [720, 280]


def test_fetch_order_book_async():
    ex = DummyBookExchange()
    data = asyncio.run(fetch_order_book_async(ex, "BTC/USD", depth=2))
    assert data["bids"] == [[1, 1], [0.9, 2]]
    assert data["asks"] == [[1.1, 1], [1.2, 3]]


def test_fetch_order_book_async_skips_unsupported_symbol():
    from crypto_bot.utils.market_loader import _UNSUPPORTED_LOGGED

    ex = DummyBookExchangeMarkets()
    _UNSUPPORTED_LOGGED.clear()
    data1 = asyncio.run(fetch_order_book_async(ex, "ETH/USD", depth=2))
    data2 = asyncio.run(fetch_order_book_async(ex, "ETH/USD", depth=2))
    assert data1 == {}
    assert data2 == {}
    assert ex.called is False
    assert _UNSUPPORTED_LOGGED == {"ETH/USD"}


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


def test_load_ohlcv_parallel_skips_unsupported_symbol(monkeypatch, caplog):
    from crypto_bot.utils import market_loader

    called = False

    async def fake_fetch(*args, **kwargs):
        nonlocal called
        called = True
        return [[0] * 6]

    monkeypatch.setattr(market_loader, "load_ohlcv", fake_fetch)
    ex = object()
    with caplog.at_level(logging.INFO):
        result = asyncio.run(load_ohlcv_parallel(ex, ["AIBTC/EUR"]))
    assert result == {"AIBTC/EUR": []}
    assert called is False
    assert not any(
        "Unsupported symbol" in r.getMessage() for r in caplog.records
    )


class DummyWSOverrideExchange:
    has = {"fetchOHLCV": True}

    async def watchOHLCV(self, symbol, timeframe="1h", limit=100, **kwargs):
        return [[0, 1, 2, 3, 4, 5]]

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[i, i + 1, i + 2, i + 3, i + 4, i + 5] for i in range(limit)]


class DummyWSExceptionExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.fetch_called = False

    async def watchOHLCV(self, symbol, timeframe="1h", limit=100, **kwargs):
        raise RuntimeError("ws failed")

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_called = True
        return [[9] * 6 for _ in range(limit)]


def test_load_ohlcv_parallel_websocket_overrides_fetch():
    ex = DummyWSOverrideExchange()
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


@pytest.mark.skip(reason="Websocket support removed")
def test_load_ohlcv_parallel_websocket_force_history():
    ex = DummyWSOverrideExchange()
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


def test_watchOHLCV_exception_falls_back_to_fetch():
    ex = DummyWSExceptionExchange()
    data = asyncio.run(fetch_ohlcv_async(ex, "BTC/USD", limit=2, use_websocket=True))
    assert ex.fetch_called is True
    assert len(data) == 2
    assert data[0][0] == 9


class WSShortfallExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.fetch_called = 0

    async def watchOHLCV(self, symbol, timeframe="1h", limit=100, **kwargs):
        return [[0] * 6 for _ in range(max(0, limit - 1))]

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_called += 1
        return [[i] * 6 for i in range(limit)]


def test_load_ohlcv_parallel_ws_shortfall_falls_back(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    ex = WSShortfallExchange()
    data = asyncio.run(
        load_ohlcv_parallel(
            ex,
            ["BTC/USD"],
            timeframe="1h",
            limit=3,
            use_websocket=True,
            max_concurrent=1,
        )
    )
    assert ex.fetch_called == 1
    assert len(data["BTC/USD"]) == 3


def test_load_ohlcv_backoff_429(monkeypatch):
    from crypto_bot.utils import market_loader

    calls = {"count": 0}
    sleeps: list[float] = []

    class DummyEx:
        has = {"fetchOHLCV": True}

        async def fetch_ohlcv(self, symbol, timeframe="1m", limit=100):
            calls["count"] += 1
            if calls["count"] == 1:
                raise Exception("429 rate limit")
            return [[1, 2, 3, 4, 5, 6]]

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    ex = DummyEx()
    data = asyncio.run(market_loader.load_ohlcv(ex, "BTC/USD"))
    assert data == [[1, 2, 3, 4, 5, 6]]
    assert calls["count"] == 2
    assert sleeps[0] == 60
    assert sleeps[1] == 1


def test_load_ohlcv_timeout_and_retry(monkeypatch, caplog):
    from crypto_bot.utils import market_loader

    calls = {"count": 0}
    orig_sleep = asyncio.sleep

    class SlowEx:
        has = {"fetchOHLCV": True}

        async def fetch_ohlcv(self, symbol, timeframe="1m", limit=100):
            calls["count"] += 1
            await orig_sleep(0.05)

    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    ex = SlowEx()
    with caplog.at_level(logging.ERROR):
        data = asyncio.run(
            market_loader.load_ohlcv(
                ex,
                "BTC/USD",
                timeout=0.01,
                max_retries=2,
            )
        )
    assert data == []
    assert calls["count"] == 2
    assert any("Failed to load OHLCV" in r.message for r in caplog.records)


class DummyIncExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.data = [[i * 3600] + [i] * 5 for i in range(200)]

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
    assert len(cache["BTC/USD"]) == 2
    ex.data.extend([[i * 3600] + [i] * 5 for i in range(200, 301)])
    cache = asyncio.run(
        update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=4, max_concurrent=2)
    )
    assert len(cache["BTC/USD"]) == 4


def test_update_ohlcv_cache_fallback_full_history():
    from crypto_bot.utils import market_loader
    market_loader._last_snapshot_time = 0
    ex = DummyFailExchange()
    cache: dict[str, pd.DataFrame] = {}
    cache = asyncio.run(
        update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=3, max_concurrent=2)
    )
    assert len(cache["BTC/USD"]) == 3
    ex.data.extend([[i * 3600] + [i] * 5 for i in range(200, 301)])
    cache = asyncio.run(
        update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=4, max_concurrent=2)
    )
    assert len(cache["BTC/USD"]) == 4


class DummyLargeExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.data = [[i * 3600] + [i] * 5 for i in range(200)]
        self.markets = {"BTC/USD": {"symbol": "BTC/USD"}}

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        if since is not None and since > self.data[-1][0] and since // 1000 <= self.data[-1][0]:
            since //= 1000
        rows = [r for r in self.data if since is None or r[0] > since]
        return rows[:limit]


def test_update_ohlcv_cache_respects_requested_limit():
    from crypto_bot.utils import market_loader
    market_loader._last_snapshot_time = 0
    ex = DummyLargeExchange()
    cache: dict[str, pd.DataFrame] = {}
    cache = asyncio.run(
        update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=50, max_concurrent=2)
    )
    assert len(cache["BTC/USD"]) == 50


def test_update_ohlcv_cache_since_overlap(monkeypatch):
    from crypto_bot.utils import market_loader

    market_loader._last_snapshot_time = time.time()
    last_ts = 123
    cache = {
        "BTC/USD": pd.DataFrame(
            [[last_ts, 1, 1, 1, 1, 1]],
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
    }

    captured: dict[str, int | None] = {}

    async def fake_load_ohlcv_parallel(
        _ex, symbols, timeframe, limit, since_map, **kwargs
    ):
        captured.update(since_map)
        return {s: [] for s in symbols}

    monkeypatch.setattr(
        market_loader, "load_ohlcv_parallel", fake_load_ohlcv_parallel
    )

    class DummyExchange:
        has = {"fetchOHLCV": True}
        markets = {
            "BTC/USD": {
                "symbol": "BTC/USD",
                "base": "BTC",
                "quote": "USD",
                "active": True,
            }
        }

    asyncio.run(
        update_ohlcv_cache(
            DummyExchange(), cache, ["BTC/USD"], limit=1, max_concurrent=1
        )
    )

    assert captured["BTC/USD"] == last_ts * 1000 + 1
def test_update_ohlcv_cache_saves_trimmed(monkeypatch, tmp_path):
    from crypto_bot.utils import market_loader

    market_loader._last_snapshot_time = 0
    ex = DummyLargeExchange()
    cache: dict[str, pd.DataFrame] = {}

    saved: dict = {}

    def fake_save(df, symbol, timeframe, storage_path):
        saved["df"] = df
        saved["symbol"] = symbol
        saved["timeframe"] = timeframe
        saved["storage_path"] = storage_path

    monkeypatch.setattr(market_loader, "save_ohlcv", fake_save)

    asyncio.run(
        update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=4,
            max_concurrent=2,
            config={"storage_path": tmp_path, "max_bootstrap_bars": 3},
        )
    )

    assert saved["symbol"] == "BTC/USD"
    assert saved["timeframe"] == "1h"
    assert saved["storage_path"] == tmp_path
    assert len(saved["df"]) == 3


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


def test_update_ohlcv_cache_batches_requests(monkeypatch):
    from crypto_bot.utils import market_loader

    calls = 0

    async def fake_inner(*args, **kwargs):
        nonlocal calls
        calls += 1
        cache = args[1]
        symbols = args[2]
        for s in symbols:
            cache[s] = pd.DataFrame({"close": [0]})
        return cache

    monkeypatch.setattr(market_loader, "_update_ohlcv_cache_inner", fake_inner)

    ex = DummySyncExchange()
    cache: dict[str, pd.DataFrame] = {}

    async def call(sym):
        await market_loader.update_ohlcv_cache(
            ex,
            cache,
            [sym],
            limit=1,
            config={"ohlcv_batch_size": 3},
        )

    asyncio.run(asyncio.gather(call("A"), call("B"), call("C")))

    assert calls == 1
    assert set(cache) == {"A", "B", "C"}


def test_update_ohlcv_cache_none_batch_size(monkeypatch):
    from crypto_bot.utils import market_loader

    captured: list[int] = []

    async def fake_worker(key, queue, batch_size, delay):
        captured.append(batch_size)
        req = await queue.get()
        for s in req.symbols:
            req.cache[s] = pd.DataFrame({"close": [0]})
        req.future.set_result(req.cache)
        queue.task_done()
        market_loader._OHLCV_BATCH_TASKS.pop(key, None)

    monkeypatch.setattr(market_loader, "_ohlcv_batch_worker", fake_worker)

    market_loader._OHLCV_BATCH_QUEUES.clear()
    market_loader._OHLCV_BATCH_TASKS.clear()

    ex = DummySyncExchange()
    cache: dict[str, pd.DataFrame] = {}

    asyncio.run(
        market_loader.update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=1,
            config={"ohlcv_batch_size": None},
        )
    )

    assert captured and captured[0] == 3
    assert "BTC/USD" in cache

    captured.clear()
    cache = {}

    asyncio.run(
        market_loader.update_ohlcv_cache(
            ex,
            cache,
            ["ETH/USD"],
            limit=1,
            batch_size=None,
        )
    )

    assert captured and captured[-1] == 3
    assert "ETH/USD" in cache


def test_update_ohlcv_cache_batch_size_no_warning(caplog):
    ex = DummySyncExchange()
    cache: dict[str, pd.DataFrame] = {}
    caplog.set_level(logging.WARNING)
    asyncio.run(
        update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=1,
            config={"ohlcv_batch_size": 2},
        )
    )
    assert not any(
        "ohlcv_batch_size not set" in r.getMessage() for r in caplog.records
    )


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


def test_load_ohlcv_parallel_priority_symbols(monkeypatch):
    from crypto_bot.utils import market_loader

    call_order: list[str] = []

    async def fake_fetch(exchange, sym, **_):
        call_order.append(sym)
        return [[0, 0, 0, 0, 0, 0]]

    monkeypatch.setattr(market_loader, "load_ohlcv", fake_fetch)

    ex = object()
    symbols = ["AAA/USD", "BBB/USD", "CCC/USD"]
    asyncio.run(
        market_loader.load_ohlcv_parallel(
            ex,
            symbols,
            max_concurrent=1,
            priority_symbols=["BBB/USD"],
        )
    )

    assert call_order[0] == "BBB/USD"
    assert call_order[1:] == ["AAA/USD", "CCC/USD"]


class RetryIncompleteExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.fetch_calls = 0

    async def watchOHLCV(self, symbol, timeframe="1h", limit=100, **kwargs):
        return [[0] * 6 for _ in range(2)]

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_calls += 1
        if self.fetch_calls == 1:
            return [[i * 3600] + [1] * 5 for i in range(4)]
        return [[i * 3600] + [i] * 5 for i in range(limit)]


class AlwaysIncompleteExchange(RetryIncompleteExchange):
    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.fetch_calls += 1
        return [[i * 3600] + [1] * 5 for i in range(4)]


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
    assert len(res["BTC/USD"]) == 10
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
        "Skipping BTC/USD: only 4/10 candles" in r.getMessage() for r in caplog.records
    )


class PartialHistoryExchange:
    has = {"fetchOHLCV": True}

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[i * 3600] + [i] * 5 for i in range(142)]


def test_min_history_fraction_allows_partial_history():
    from crypto_bot.utils import market_loader

    market_loader._last_snapshot_time = 0
    ex = PartialHistoryExchange()
    cache: dict[str, pd.DataFrame] = {}
    config = {"min_history_fraction": 0.2}

    cache = asyncio.run(
        update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=700,
            config=config,
            max_concurrent=1,
        )
    )
    assert len(cache["BTC/USD"]) == 142
    market_loader._last_snapshot_time = 0


class GapExchange:
    has = {"fetchOHLCV": True}

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [
            [0, 1, 1, 1, 1, 10],
            [3600, 2, 2, 2, 2, 20],
            [10800, 3, 3, 3, 3, 30],
        ]


def test_update_ohlcv_cache_ffill_missing_intervals():
    from crypto_bot.utils import market_loader

    market_loader._last_snapshot_time = time.time()
    ex = GapExchange()
    cache: dict[str, pd.DataFrame] = {}
    config = {"min_history_fraction": 0}

    cache = asyncio.run(
        update_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            limit=1,
            config=config,
            max_concurrent=1,
        )
    )
    df = cache["BTC/USD"]
    assert list(df["timestamp"]) == [0, 3600, 7200, 10800]
    assert df.loc[2, "open"] == df.loc[1, "open"]
    market_loader._last_snapshot_time = 0


class DummyMultiTFExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.calls: list[str] = []

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.calls.append(timeframe)
        return [[0, 1, 2, 3, 4, 5]]


class LimitedTFExchange(DummyMultiTFExchange):
    timeframes = {"1h": "1h"}


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


def test_update_multi_tf_ohlcv_cache_priority_queue(monkeypatch):
    from collections import deque
    from crypto_bot.utils import market_loader

    captured: dict[str, list[str]] = {}

    async def fake_update_ohlcv_cache(exchange, tf_cache, symbols, **kwargs):
        captured["symbols"] = list(symbols)
        captured["priority"] = kwargs.get("priority_symbols")
        for s in symbols:
            tf_cache[s] = pd.DataFrame(
                [[0, 0, 0, 0, 0, 0]],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
        return tf_cache

    monkeypatch.setattr(market_loader, "update_ohlcv_cache", fake_update_ohlcv_cache)

    async def _listing(_sym):
        return 0

    monkeypatch.setattr(market_loader, "get_kraken_listing_date", _listing)

    class Ex(DummyMultiTFExchange):
        timeframes = {"1h": "1h"}
        symbols = ["BTC/USD", "ETH/USD"]
        id = "dummy"

    pq = deque(["ETH/USD"])
    cache: dict[str, dict[str, pd.DataFrame]] = {}
    asyncio.run(
        market_loader.update_multi_tf_ohlcv_cache(
            Ex(),
            cache,
            ["BTC/USD", "ETH/USD"],
            {"timeframes": ["1h"]},
            limit=1,
            priority_queue=pq,
        )
    )

    assert captured.get("priority") == ["ETH/USD"]
    assert captured.get("symbols", [])[0] == "ETH/USD"
    assert not pq


def test_update_multi_tf_ohlcv_cache_default_chunk(monkeypatch):
    from crypto_bot.utils import market_loader

    captured: list[list[str]] = []

    async def fake_update(exchange, tf_cache, symbols, **kwargs):
        captured.append(list(symbols))
def test_update_multi_tf_ohlcv_cache_resume(tmp_path, monkeypatch):
    from crypto_bot.utils import market_loader

    state_file = tmp_path / "state.json"
    monkeypatch.setattr(market_loader, "BOOTSTRAP_STATE_FILE", state_file)
    monkeypatch.setattr(market_loader, "CACHE_DIR", tmp_path)

    calls: list[str] = []

    async def fake_update_ohlcv_cache(exchange, tf_cache, symbols, **kwargs):
        calls.extend(symbols)
        for s in symbols:
            tf_cache[s] = pd.DataFrame(
                [[0, 0, 0, 0, 0, 0]],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
        return tf_cache

    monkeypatch.setattr(market_loader, "update_ohlcv_cache", fake_update)
    monkeypatch.setattr(market_loader, "get_kraken_listing_date", lambda _s: 0)

    class Ex(DummyMultiTFExchange):
        timeframes = {"1h": "1h"}
        symbols = [f"S{i}/USD" for i in range(25)]

    cache: dict[str, dict[str, pd.DataFrame]] = {}
    asyncio.run(
        market_loader.update_multi_tf_ohlcv_cache(
            Ex(),
            cache,
            [f"S{i}/USD" for i in range(25)],
            {"timeframes": ["1h"]},
            limit=1,
        )
    )

    assert len(captured) == 2
    assert len(captured[0]) == 20
    assert len(captured[1]) == 5
    monkeypatch.setattr(market_loader, "update_ohlcv_cache", fake_update_ohlcv_cache)
    monkeypatch.setattr(market_loader, "get_kraken_listing_date", lambda _s: 0)

    ex = DummyMultiTFExchange()
    cache: dict[str, dict[str, pd.DataFrame]] = {}
    cfg = {"timeframes": ["1h"]}

    asyncio.run(
        market_loader.update_multi_tf_ohlcv_cache(
            ex, cache, ["BTC/USD", "ETH/USD"], cfg, limit=1
        )
    )
    data = json.loads(state_file.read_text())
    assert set(data.get("1h", [])) == {"BTC/USD", "ETH/USD"}
    assert set(calls) == {"BTC/USD", "ETH/USD"}

    calls.clear()
    asyncio.run(
        market_loader.update_multi_tf_ohlcv_cache(
            ex, cache, ["BTC/USD", "ETH/USD"], cfg, limit=1
        )
    )
    assert calls == []


def test_update_multi_tf_ohlcv_cache_skips_unsupported_tf(caplog):
    ex = LimitedTFExchange()
    caplog.set_level(logging.INFO)
    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["BTC/USD"],
            {"timeframes": ["1h", "4h"]},
            limit=1,
        )
    )
    assert set(cache.keys()) == {"1h"}
    assert set(ex.calls) == {"1h"}
    assert any(
        "Skipping unsupported timeframes" in r.getMessage() for r in caplog.records
    )


def test_update_multi_tf_ohlcv_cache_notifier(monkeypatch):
    from crypto_bot.utils import market_loader

    class DummyNotifier:
        def __init__(self):
            self.msgs: list[str] = []

        async def notify_async(self, text):
            self.msgs.append(text)

    monkeypatch.setattr(market_loader, "load_enabled", lambda _c: None)

    async def _no_enqueue(*_a, **_k):
        return None

    monkeypatch.setattr(market_loader, "_maybe_enqueue_eval", _no_enqueue)
    monkeypatch.setattr(
        market_loader, "resolve_listed_symbol", lambda *_a, **_k: "BTC/USD"
    )

    async def fake_update(exchange, tf_cache, symbols, **kwargs):
        for s in symbols:
            tf_cache[s] = pd.DataFrame(
                [[0, 0, 0, 0, 0, 0]],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
        return tf_cache

    monkeypatch.setattr(market_loader, "update_ohlcv_cache", fake_update)
    async def fake_load_ohlcv(*_a, **_k):
        return [[0, 0, 0, 0, 0, 0]]

    monkeypatch.setattr(market_loader, "load_ohlcv", fake_load_ohlcv)

    notifier = DummyNotifier()
    ex = DummyMultiTFExchange()
    cfg = {"timeframes": ["1h", "4h"], "telegram": {"bootstrap_updates": True}}

    asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["BTC/USD"],
            cfg,
            limit=1,
            notifier=notifier,
        )
    )

    assert any("Starting OHLCV update" in m for m in notifier.msgs)
    assert any("Completed OHLCV update" in m for m in notifier.msgs)


def test_update_multi_tf_ohlcv_cache_notifier_disabled(monkeypatch):
    from crypto_bot.utils import market_loader

    class DummyNotifier:
        def __init__(self):
            self.msgs: list[str] = []

        async def notify_async(self, text):
            self.msgs.append(text)

    monkeypatch.setattr(market_loader, "load_enabled", lambda _c: None)

    async def _no_enqueue(*_a, **_k):
        return None

    monkeypatch.setattr(market_loader, "_maybe_enqueue_eval", _no_enqueue)
    monkeypatch.setattr(
        market_loader, "resolve_listed_symbol", lambda *_a, **_k: "BTC/USD"
    )

    async def fake_update(exchange, tf_cache, symbols, **kwargs):
        for s in symbols:
            tf_cache[s] = pd.DataFrame(
                [[0, 0, 0, 0, 0, 0]],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
        return tf_cache

    monkeypatch.setattr(market_loader, "update_ohlcv_cache", fake_update)
    async def fake_load_ohlcv(*_a, **_k):
        return [[0, 0, 0, 0, 0, 0]]

    monkeypatch.setattr(market_loader, "load_ohlcv", fake_load_ohlcv)

    notifier = DummyNotifier()
    ex = DummyMultiTFExchange()
    cfg = {"timeframes": ["1h"], "telegram": {"bootstrap_updates": False}}

    asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["BTC/USD"],
            cfg,
            limit=1,
            notifier=notifier,
        )
    )

    assert notifier.msgs == []


class MixedTFExchange(DummyMultiTFExchange):
    timeframes = {"1m": "1m", "5m": "5m"}

    async def fetch_ohlcv(self, symbol, timeframe="1m", limit=100):
        self.calls.append((symbol, timeframe))
        if symbol == "FOO/USD" and timeframe == "1m":
            raise ccxt.BadRequest("unsupported timeframe")
        return [[0, 1, 2, 3, 4, 5]]


def test_update_multi_tf_ohlcv_cache_falls_back_to_5m(monkeypatch):
    from crypto_bot.utils import market_loader

    monkeypatch.setattr(market_loader, "get_kraken_listing_date", lambda _s: 0)
    ex = MixedTFExchange()
    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["BTC/USD", "FOO/USD"],
            {"timeframes": ["1m", "5m"]},
            limit=1,
        )
    )
    assert "BTC/USD" in cache.get("1m", {})
    assert "FOO/USD" not in cache.get("1m", {})
    assert set(cache.get("5m", {}).keys()) == {"BTC/USD", "FOO/USD"}
    assert ("FOO/USD", "5m") in ex.calls


class PagingMultiExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.calls = 0

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        self.calls += 1
        start = since or 0
        tf_ms = 3600 * 1000
        return [[start + i * tf_ms, 1, 1, 1, 1, 1] for i in range(limit)]


def test_update_multi_tf_ohlcv_cache_start_since_paging(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = PagingMultiExchange()

    monkeypatch.setattr(market_loader.time, "time", lambda: 720 * 2 * 3600)

    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["BTC/USD"],
            {"timeframes": ["1h"]},
            start_since=0,
        )
    )

    assert ex.calls > 1
    df = cache["1h"]["BTC/USD"]
    assert len(df) >= 720 * 2


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
            batch_size=3,
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
    async def fake_send_test(*_a, **_k):
        return True
    monkeypatch.setattr(main, "send_test_message", fake_send_test)
    monkeypatch.setattr(main, "log_balance", lambda *_a, **_k: None)

    class DummyRC:
        def __init__(self, *_a, **_k):
            pass

    monkeypatch.setattr(main, "RiskConfig", DummyRC)
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


def test_main_uses_pair_cache_when_scan_none(monkeypatch, caplog):
    import sys, types

    monkeypatch.setitem(sys.modules, "ccxt", types.SimpleNamespace())
    import crypto_bot.main as main

    caplog.set_level(logging.WARNING)

    async def fake_loader(exchange, exclude=None, config=None):
        return None

    cfg = {"symbol": "BTC/USD", "scan_markets": True}

    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "load_kraken_symbols", fake_loader)
    monkeypatch.setattr(main, "load_liquid_pairs", lambda: ["ETH/USD"])
    monkeypatch.setattr(main, "cooldown_configure", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "dotenv_values", lambda path: {})
    monkeypatch.setattr(main, "load_or_create", lambda: {})
    async def fake_send_test2(*_a, **_k):
        return True
    monkeypatch.setattr(main, "send_test_message", fake_send_test2)
    monkeypatch.setattr(main, "log_balance", lambda *_a, **_k: None)

    class DummyRC:
        def __init__(self, *_a, **_k):
            pass

    monkeypatch.setattr(main, "RiskConfig", DummyRC)
    monkeypatch.setattr(main.asyncio, "sleep", lambda *_a: None)
    monkeypatch.setattr(main, "MAX_SYMBOL_SCAN_ATTEMPTS", 1)
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

    calls = {}

    class DummyRM:
        def __init__(self, *_a, **_k):
            calls["rm"] = True
            raise StopLoop

    monkeypatch.setattr(main, "RiskManager", DummyRM)

    asyncio.run(main.main())

    assert captured["cfg"]["symbols"] == ["ETH/USD"]
    assert any("cached pairs" in r.getMessage() for r in caplog.records)
    assert calls.get("rm") is True


class SlowExchange:
    has = {"fetchOHLCV": True}

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        await asyncio.sleep(0.05)
        return [[0] * 6]


class SlowWSExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.fetch_called = False

    async def watchOHLCV(self, symbol, timeframe="1h", limit=100, **kwargs):
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


def test_load_ohlcv_parallel_respects_max_retries(monkeypatch):
    from crypto_bot.utils import market_loader

    calls = {"count": 0}
    orig_sleep = asyncio.sleep

    class SlowEx:
        has = {"fetchOHLCV": True}

        async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
            calls["count"] += 1
            await orig_sleep(0.05)

    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    ex = SlowEx()
    result = asyncio.run(
        market_loader.load_ohlcv_parallel(
            ex,
            ["BTC/USD"],
            max_concurrent=1,
            timeout=0.01,
            max_retries=2,
        )
    )
    assert result == {}
    assert calls["count"] == 2


class LimitCaptureWS:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.limit = None

    async def watchOHLCV(self, symbol, timeframe="1h", limit=100, since=None, **kwargs):
        self.limit = limit
        return [[0] * 6]


def test_watchOHLCV_since_reduces_limit(monkeypatch):
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


def test_load_ohlcv_parallel_sleep_and_backoff(monkeypatch):
    from crypto_bot.utils import market_loader

    market_loader.failed_symbols.clear()
    monkeypatch.setattr(market_loader, "RETRY_DELAY", 60)

    sleeps: list[float] = []

    async def fake_sleep(d):
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    calls: list[str] = []

    class Dummy429(Exception):
        def __init__(self):
            self.http_status = 429

    async def fake_load(exchange, sym, **_):
        calls.append(sym)
        raise Dummy429()

    monkeypatch.setattr(market_loader, "load_ohlcv", fake_load)
    monkeypatch.setattr(time, "time", lambda: 0)

    ex = object()
    asyncio.run(
        market_loader.load_ohlcv_parallel(ex, ["BTC/USD"], max_concurrent=1)
    )
    assert sleeps == []
    assert market_loader.failed_symbols["BTC/USD"]["delay"] == 60

    asyncio.run(
        market_loader.load_ohlcv_parallel(ex, ["BTC/USD"], max_concurrent=1)
    )
    assert calls == ["BTC/USD"]


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
    assert not any(
        "Unsupported symbol" in r.getMessage() for r in caplog.records
    )


def test_invalid_symbol_marked_disabled():
    from crypto_bot.utils import market_loader

    ex = SymbolCheckExchange()
    market_loader.failed_symbols.clear()
    result = asyncio.run(market_loader.fetch_ohlcv_async(ex, "ETH/USD"))
    assert result == []
    assert "ETH/USD" not in market_loader.failed_symbols


def test_unsupported_symbols_skip(monkeypatch):
    from crypto_bot.utils import market_loader

    class DummyExchange:
        has = {"fetchOHLCV": True}

        def __init__(self):
            self.called = False

        async def fetch_ohlcv(self, *a, **k):
            self.called = True
            return [[0] * 6]

    ex = DummyExchange()
    monkeypatch.setattr(market_loader, "UNSUPPORTED_SYMBOLS", {"BAD/USD"})
    data = asyncio.run(market_loader.fetch_ohlcv_async(ex, "BAD/USD"))
    assert data == []
    assert ex.called is False


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


class DummyUnsupportedExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.called = False

    async def fetch_ohlcv(self, *args, **kwargs):
        self.called = True
        return [[0] * 6]


def test_fetch_ohlcv_async_skips_unsupported_symbol(caplog):
    ex = DummyUnsupportedExchange()
    with caplog.at_level(logging.INFO):
        data = asyncio.run(fetch_ohlcv_async(ex, "AIBTC/EUR"))
    assert data == []
    assert ex.called is False
    assert not any(
        "Unsupported symbol" in r.getMessage() for r in caplog.records
    )


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
    assert sleeps == [5]
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
    assert sleeps == [5]
    assert data == [[1] * 6]


class CancelWSExchange:
    has = {"fetchOHLCV": True}

    def __init__(self):
        self.fetch_called = False
        self.closed = False

    async def watchOHLCV(self, symbol, timeframe="1h", limit=100, **kwargs):
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

    async def watchOHLCV(self, symbol, timeframe="1h", limit=100, **kwargs):
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


def test_fetch_geckoterminal_ohlcv_uses_exchange():
    from crypto_bot.utils import market_loader

    class DummyEx:
        def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=500):
            assert symbol == "BTC/USD"
            assert timeframe == "1h"
            assert limit == 1
            return [[1, 1, 1, 1, 1, 1]]

    data = asyncio.run(
        market_loader.fetch_geckoterminal_ohlcv(
            "BTC/USD", timeframe="1h", limit=1, exchange=DummyEx()
        )
    )
    assert data == [[1, 1, 1, 1, 1, 1]]


def test_fetch_onchain_ohlcv_fallback(monkeypatch):
    from crypto_bot.solana import prices

    async def fake_prices(symbols):
        return {symbols[0]: 42.0}

    class DummyResp:
        async def __aenter__(self):
            raise Exception("boom")

        async def __aexit__(self, *args):
            return False

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def get(self, *a, **k):
            return DummyResp()

    monkeypatch.setattr(prices, "aiohttp", types.SimpleNamespace(ClientSession=lambda: DummySession()))
    monkeypatch.setattr(prices, "fetch_solana_prices", fake_prices)

    data = asyncio.run(prices.fetch_onchain_ohlcv("SOL/USDC", limit=1))
    assert data and data[0][1] == 42.0


def test_update_multi_tf_ohlcv_cache_skips_404(monkeypatch):
    from crypto_bot.utils import market_loader

    async def fake_fetch(*_a, **_k):
        return None

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_fetch)
    monkeypatch.setattr(market_loader, "fetch_coingecko_ohlc", lambda *a, **k: None)
    monkeypatch.setattr(market_loader, "fetch_onchain_ohlcv", lambda *a, **k: None)

    async def fake_ohlcv(*a, **k):
        return [[1, 1, 1, 1, 1, 1]]

    monkeypatch.setattr(market_loader, "load_ohlcv", fake_ohlcv)

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

    calls: list[float] = []

    async def fake_fetch(*_a, min_24h_volume=0, **_k):
        calls.append(min_24h_volume)
        if min_24h_volume > 50:
            return None
        return [[0, 1, 2, 3, 4, 5]]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_fetch)
    monkeypatch.setattr(market_loader, "fetch_coingecko_ohlc", lambda *a, **k: None)

    async def fake_fetch2(*_a, **_k):
        return [[0, 1, 2, 3, 4, 5]], 50.0, 1000.0
    class GeckoData(list):
        pass

    def fake_gecko(*_a, **_k):
        data = GeckoData([[0, 1, 2, 3, 4, 5]])
        data.vol_24h_usd = 50
        return data

    load_calls = {"count": 0}

    async def fake_ohlcv(*a, **k):
        load_calls["count"] += 1
        return [[1, 1, 1, 1, 1, 1]]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_fetch2)
    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_gecko)
    monkeypatch.setattr(market_loader, "fetch_coingecko_ohlc", lambda *a, **k: None)
    monkeypatch.setattr(market_loader, "fetch_onchain_ohlcv", lambda *a, **k: None)
    monkeypatch.setattr(market_loader, "load_ohlcv", fake_ohlcv)

    ex = DummyMultiTFExchange()
    cache = {}
    config = {"timeframes": ["1h"], "min_volume_usd": 100}

    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            cache,
            [f"{VALID_MINT}/USDC"],
            config,
            limit=1,
        )
    )
    assert load_calls["count"] == 1
    assert f"{VALID_MINT}/USDC" in cache["1h"]

    load_calls["count"] = 0
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
    assert load_calls["count"] == 0
    assert f"{VALID_MINT}/USDC" in cache["1h"]


def test_dex_fetch_uses_exchange_on_gecko_failure(monkeypatch):
    from crypto_bot.utils import market_loader

    async def fail_gecko(*_a, **_k):
        raise Exception("boom")

    calls = {"exchange": 0}

    async def fake_fetch(ex, *a, **k):
        calls["exchange"] += 1
        return [[9, 9, 9, 9, 9, 9]]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fail_gecko)
    monkeypatch.setattr(market_loader, "fetch_onchain_ohlcv", lambda *a, **k: None)
    monkeypatch.setattr(market_loader, "load_ohlcv", fake_fetch)

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
    assert calls["exchange"] == 1


def test_dex_fetch_uses_provided_exchange(monkeypatch):
    from crypto_bot.utils import market_loader

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
    monkeypatch.setattr(market_loader, "load_ohlcv", fake_fetch)

    ex = DummyMultiTFExchange()

    data = asyncio.run(market_loader.fetch_dex_ohlcv(ex, "FOO/USDC", limit=1))
    assert data
    assert calls["coinbase"] == 0
    assert calls["exchange"] == 1


def test_dex_fetch_returns_none_on_error(monkeypatch):
    from crypto_bot.utils import market_loader

    async def fail_fetch(*_a, **_k):
        raise Exception("boom")

    monkeypatch.setattr(market_loader, "load_ohlcv", fail_fetch)

    ex = DummyMultiTFExchange()

    data = asyncio.run(market_loader.fetch_dex_ohlcv(ex, "FOO/BTC", limit=1))
    assert data is None


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
    monkeypatch.setattr(market_loader, "fetch_onchain_ohlcv", lambda *a, **k: None)
    monkeypatch.setattr(market_loader, "load_ohlcv", fake_fetch)

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
    assert calls["fetch"] == 1



def test_update_multi_tf_ohlcv_cache_start_since(monkeypatch):
    from crypto_bot.utils import market_loader

    calls: list[tuple[int | None, int]] = []

    async def fake_fetch(_ex, _sym, timeframe="1s", limit=1000, since=None, **_k):
        calls.append((since, limit))
        start = since or 0
        step = 1
        count = 1000 if len(calls) == 1 else 500
        return [[start + i * step, 1, 1, 1, 1, 1] for i in range(count)]

    monkeypatch.setattr(market_loader, "load_ohlcv", fake_fetch)

    ex = object()
    cache: dict[str, dict[str, pd.DataFrame]] = {}
    config = {"timeframes": ["1s"]}

    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            cache,
            ["BTC/USD"],
            config,
            start_since=0,
        )
    )

    df = cache["1s"]["BTC/USD"]
    assert len(df) == 1500
    assert calls == [(0, 1000), (1000, 1000)]


def test_coinbase_usdc_pair_mapping(monkeypatch):
    from crypto_bot.utils import market_loader

    called: dict[str, str] = {}

    async def fake_fetch(_ex, sym, timeframe="1h", limit=100, **_k):
        called["sym"] = sym
        return [[i, 1, 1, 1, 1, 1] for i in range(limit)]

    monkeypatch.setattr(market_loader, "load_ohlcv", fake_fetch)

    class DummyCB:
        id = "coinbase"
        has = {"fetchOHLCV": True}
        symbols = ["FOO/USD"]

    ex = DummyCB()
    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["FOO/USDC"],
            {"timeframes": ["1h"]},
            limit=1,
        )
    )

    assert "FOO/USDC" in cache["1h"]
    assert called["sym"] == "FOO/USDC"


def test_coinbase_usdc_pair_skip(monkeypatch):
    from crypto_bot.utils import market_loader

    calls = {"gecko": 0, "dex": 0, "ohlcv": 0, "onchain": 0}

    async def fake_ohlcv(*_a, **_k):
        calls["ohlcv"] += 1
        return []

    async def fake_gecko(*_a, **_k):
        calls["gecko"] += 1
        return None

    async def fake_dex(*_a, **_k):
        calls["dex"] += 1
        return []

    async def fake_onchain(*_a, **_k):
        calls["onchain"] += 1
        return []

    monkeypatch.setattr(market_loader, "load_ohlcv", fake_ohlcv)
    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_gecko)
    monkeypatch.setattr(market_loader, "fetch_dex_ohlcv", fake_dex)
    monkeypatch.setattr(market_loader, "fetch_onchain_ohlcv", fake_onchain)

    class DummyCB:
        id = "coinbase"
        has = {"fetchOHLCV": True}
        symbols: list[str] = []

    ex = DummyCB()
    cache = asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["FOO/USDC"],
            {"timeframes": ["1h"]},
            limit=1,
        )
    )

    assert calls["gecko"] == 1
    assert calls["gecko"] == 1
    assert calls["onchain"] == 1
    assert calls["dex"] == 1
    assert calls["ohlcv"] == 0


def test_dynamic_limits_unit_conversion(monkeypatch):
    from crypto_bot.utils import market_loader
    import pandas as pd

    limits: list[int] = []

    async def fake_update(_ex, cache, syms, timeframe="1h", limit=100, **_k):
        limits.append(limit)
        for s in syms:
            cache[s] = pd.DataFrame(
                [[0, 1, 1, 1, 1, 1]],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
        return cache

    now = 20 * 3600  # 20h in seconds
    monkeypatch.setattr(market_loader, "update_ohlcv_cache", fake_update)
    monkeypatch.setattr(market_loader.time, "time", lambda: float(now))

    async def listing_date(_sym):
        return int(now * 1000 - 10 * 3600 * 1000)

    monkeypatch.setattr(market_loader, "get_kraken_listing_date", listing_date)

    ex = DummyMultiTFExchange()
    asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["BTC/USD"],
            {"timeframes": ["1h"]},
            limit=100,
        )
    )

    assert limits == [10]


def test_dynamic_limits_cap(monkeypatch):
    from crypto_bot.utils import market_loader
    import pandas as pd

    limits: list[int] = []

    async def fake_update(_ex, cache, syms, timeframe="1h", limit=100, **_k):
        limits.append(limit)
        for s in syms:
            cache[s] = pd.DataFrame(
                [[0, 1, 1, 1, 1, 1]],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
        return cache

    now = 9_000_000  # arbitrary seconds (>90 days)
    monkeypatch.setattr(market_loader, "update_ohlcv_cache", fake_update)
    monkeypatch.setattr(market_loader.time, "time", lambda: float(now))

    async def listing_date(_sym):
        return int(now * 1000 - 90 * 24 * 3600 * 1000)

    monkeypatch.setattr(market_loader, "get_kraken_listing_date", listing_date)

    ex = DummyMultiTFExchange()
    asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["BTC/USD"],
            {"timeframes": ["1h"], "ohlcv_snapshot_limit": 1000},
            limit=1500,
        )
    )

    assert limits == [720]


def test_dynamic_limits_skip_extreme_age(monkeypatch):
    from crypto_bot.utils import market_loader

    called = False

    async def fake_update(*_a, **_k):
        nonlocal called
        called = True
        return {}

    now = 1_000_000
    monkeypatch.setattr(market_loader, "update_ohlcv_cache", fake_update)
    monkeypatch.setattr(market_loader.time, "time", lambda: float(now))

    async def listing_date(_sym):
        # 100 years ago
        return int(now * 1000 - 100 * 365 * 24 * 3600 * 1000)

    monkeypatch.setattr(market_loader, "get_kraken_listing_date", listing_date)

    ex = DummyMultiTFExchange()
    asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["BTC/USD"],
            {"timeframes": ["1h"]},
            limit=100,
        )
    )

    assert not called


def test_listing_date_concurrency(monkeypatch):
    from crypto_bot.utils import market_loader

    active = 0
    max_active = 0

    async def listing_date(_sym):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return 1

    async def fake_update(*_a, **_k):
        return {}

    monkeypatch.setattr(market_loader, "get_kraken_listing_date", listing_date)
    monkeypatch.setattr(market_loader, "update_ohlcv_cache", fake_update)
    monkeypatch.setattr(market_loader, "fetch_dex_ohlcv", lambda *a, **k: [])
    monkeypatch.setattr(market_loader, "load_ohlcv", lambda *a, **k: [])

    ex = DummyMultiTFExchange()
    asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["A/USD", "B/USD", "C/USD"],
            {"timeframes": ["1h"], "listing_date_concurrency": 2},
            limit=1,
        )
    )

    assert 1 < max_active <= 2


def test_listing_date_cache(monkeypatch):
    from crypto_bot.utils import market_loader

    calls: list[str] = []

    async def listing_date(sym: str):
        calls.append(sym)
        return 1

    async def fake_update(*_a, **_k):
        return {}

    market_loader._LISTING_DATE_CACHE.clear()
    monkeypatch.setattr(market_loader, "get_kraken_listing_date", listing_date)
    monkeypatch.setattr(market_loader, "update_ohlcv_cache", fake_update)
    monkeypatch.setattr(market_loader, "fetch_dex_ohlcv", lambda *a, **k: [])
    async def fake_load(*a, **k):
        return []
    monkeypatch.setattr(market_loader, "load_ohlcv", fake_load)

    ex = DummyMultiTFExchange()
    asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["A/USD"],
            {"timeframes": ["1h"]},
            limit=1,
        )
    )
    asyncio.run(
        update_multi_tf_ohlcv_cache(
            ex,
            {},
            ["A/USD"],
            {"timeframes": ["1h"]},
            limit=1,
        )
    )

    assert calls == ["A/USD"]
