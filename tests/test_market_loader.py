import asyncio
import pandas as pd
import pytest

from crypto_bot.utils.market_loader import (
    load_kraken_symbols,
    fetch_ohlcv_async,
    load_ohlcv_parallel,
    update_ohlcv_cache,
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


class DummyAsyncExchange:
    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[0, 1, 2, 3, 4, 5]]


class DummyWSExchange:
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


class DummySyncExchange:
    def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[0, 1, 2, 3, 4, 5]]


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
    async def watch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[0, 1, 2, 3, 4, 5]]

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        return [[i, i + 1, i + 2, i + 3, i + 4, i + 5] for i in range(limit)]


class DummyWSExceptionExchange:
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


class DummyIncExchange:
    def __init__(self):
        self.data = [[i] * 6 for i in range(1, 4)]

    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        rows = [r for r in self.data if since is None or r[0] > since]
        return rows[:limit]


class DummyFailExchange(DummyIncExchange):
    async def fetch_ohlcv(self, symbol, timeframe="1h", since=None, limit=100):
        if since is not None:
            return []
        return await super().fetch_ohlcv(symbol, timeframe, since, limit)


def test_update_ohlcv_cache_appends():
    ex = DummyIncExchange()
    cache: dict[str, pd.DataFrame] = {}
    cache = asyncio.run(update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=2, max_concurrent=2))
    assert len(cache["BTC/USD"]) == 2
    ex.data.append([4] * 6)
    cache = asyncio.run(update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=4, max_concurrent=2))
    assert len(cache["BTC/USD"]) == 4


def test_update_ohlcv_cache_fallback_full_history():
    ex = DummyFailExchange()
    cache: dict[str, pd.DataFrame] = {}
    cache = asyncio.run(update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=3, max_concurrent=2))
    assert len(cache["BTC/USD"]) == 3
    ex.data.append([4] * 6)
    cache = asyncio.run(update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=4, max_concurrent=2))
    assert len(cache["BTC/USD"]) == 4
class CountingExchange:
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


class FailOnceExchange:
    def __init__(self):
        self.calls = 0

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("fail")
        return [[0] * 6]


def test_failed_symbol_skipped_until_delay(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = FailOnceExchange()
    cache: dict[str, pd.DataFrame] = {}
    market_loader.failed_symbols.clear()
    monkeypatch.setattr(market_loader, "retry_delay", 10)

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
