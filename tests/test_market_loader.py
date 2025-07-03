import asyncio

from crypto_bot.utils.market_loader import (
    load_kraken_symbols,
    fetch_ohlcv_async,
    load_ohlcv_parallel,
)

class DummyExchange:
    def load_markets(self):
        return {
            "BTC/USD": {"active": True},
            "ETH/USD": {"active": True},
            "XRP/USD": {"active": False},
        }

def test_load_kraken_symbols_returns_active():
    ex = DummyExchange()
    symbols = asyncio.run(load_kraken_symbols(ex))
    assert set(symbols) == {"BTC/USD", "ETH/USD"}

def test_excluded_symbols_are_removed():
    ex = DummyExchange()
    symbols = asyncio.run(load_kraken_symbols(ex, exclude=["ETH/USD"]))
    assert set(symbols) == {"BTC/USD"}


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
        load_ohlcv_parallel(ex, ["BTC/USD", "ETH/USD"], timeframe="1h", limit=1)
    )
    assert set(result.keys()) == {"BTC/USD", "ETH/USD"}
