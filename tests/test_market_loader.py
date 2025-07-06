import asyncio
import pandas as pd
import pytest
import logging

from crypto_bot.utils.market_loader import (
    load_kraken_symbols,
    fetch_ohlcv_async,
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
        self.data = [[i] * 6 for i in range(100, 400, 100)]

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
    ex.data.append([400] * 6)
    cache = asyncio.run(update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=4, max_concurrent=2))
    assert len(cache["BTC/USD"]) == 4


def test_update_ohlcv_cache_fallback_full_history():
    ex = DummyFailExchange()
    cache: dict[str, pd.DataFrame] = {}
    cache = asyncio.run(update_ohlcv_cache(ex, cache, ["BTC/USD"], limit=3, max_concurrent=2))
    assert len(cache["BTC/USD"]) == 3
    ex.data.append([400] * 6)
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


class DummyMultiTFExchange:
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
    cache = asyncio.run(
        update_regime_tf_cache(
            ex,
            cache,
            ["BTC/USD"],
            config,
            limit=1,
            max_concurrent=2,
        )
    )
    assert set(cache.keys()) == {"5m", "15m", "1h"}
    for tf in config["regime_timeframes"]:
        assert "BTC/USD" in cache[tf]
    assert set(ex.calls) == {"5m", "15m", "1h"}
class FailOnceExchange:
    def __init__(self):
        self.calls = 0

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("fail")
        return [[0] * 6]


class AlwaysFailExchange:
    def __init__(self):
        self.calls = 0

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        self.calls += 1
        raise RuntimeError("fail")


class FailSuccessExchange:
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


def test_failed_symbol_backoff(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = AlwaysFailExchange()
    cache: dict[str, pd.DataFrame] = {}
    market_loader.failed_symbols.clear()
    monkeypatch.setattr(market_loader, "retry_delay", 10)
    monkeypatch.setattr(market_loader, "max_retry_delay", 40)

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


class StopLoop(Exception):
    pass


def test_main_preserves_symbols_on_scan_failure(monkeypatch, caplog):
    import crypto_bot.main as main

    caplog.set_level(logging.WARNING)

    async def fake_loader(exchange, exclude=None, config=None):
        main.logger.warning("symbol scan empty")
        return []

    cfg = {"symbol": "BTC/USD", "scan_markets": True}

    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "load_kraken_symbols", fake_loader)
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

    captured = {}

    class DummyExchange:
        def fetch_balance(self):
            return {"USDT": {"free": 0}}

    def fake_get_exchange(config):
        captured["cfg"] = config
        return DummyExchange(), None

    monkeypatch.setattr(main, "get_exchange", fake_get_exchange)

    with pytest.raises(StopLoop):
        asyncio.run(main.main())

    assert "symbols" not in captured["cfg"]
    assert any("symbol scan empty" in r.getMessage() for r in caplog.records)


class SlowExchange:
    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        await asyncio.sleep(0.05)
        return [[0] * 6]


class SlowWSExchange:
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

    res = asyncio.run(market_loader.fetch_ohlcv_async(ex, "BTC/USD"))
    assert isinstance(res, asyncio.TimeoutError)


def test_load_ohlcv_parallel_timeout(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = SlowExchange()
    market_loader.failed_symbols.clear()
    monkeypatch.setattr(market_loader, "OHLCV_TIMEOUT", 0.01)

    result = asyncio.run(
        market_loader.load_ohlcv_parallel(ex, ["BTC/USD"], max_concurrent=1)
    )
    assert result == {}
    assert "BTC/USD" in market_loader.failed_symbols


def test_fetch_ohlcv_async_timeout_fallback(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = SlowWSExchange()
    monkeypatch.setattr(market_loader, "OHLCV_TIMEOUT", 0.01)

    data = asyncio.run(
        market_loader.fetch_ohlcv_async(
            ex, "BTC/USD", limit=2, use_websocket=True
        )
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


class LimitCaptureExchange:
    def __init__(self):
        self.limit = None

    async def watch_ohlcv(self, symbol, timeframe="1h", limit=100, since=None):
        self.limit = limit
        return [[0] * 6]


def test_watch_ohlcv_since_reduces_limit(monkeypatch):
    from crypto_bot.utils import market_loader

    ex = LimitCaptureExchange()
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
