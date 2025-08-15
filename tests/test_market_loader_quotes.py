import asyncio
import logging

from crypto_bot.utils.market_loader import load_kraken_symbols, load_ohlcv, logger as ml_logger


class QuoteExchange:
    exchange_market_types = {"spot"}

    def load_markets(self):
        return {
            "BTC/USD": {"active": True, "type": "spot", "quote": "USD"},
            "BTC/EUR": {"active": True, "type": "spot", "quote": "EUR"},
            "BTC/USDT": {"active": True, "type": "spot", "quote": "USDT"},
            "BTC/JPY": {"active": True, "type": "spot", "quote": "JPY"},
        }


def test_load_kraken_symbols_filters_quotes():
    ex = QuoteExchange()
    symbols = asyncio.run(load_kraken_symbols(ex))
    assert set(symbols) == {"BTC/USD", "BTC/EUR", "BTC/USDT"}


class SyntheticExchange:
    exchange_market_types = {"spot"}

    def load_markets(self):
        return {
            "BTC/USD": {"active": True, "type": "spot", "quote": "USD"},
            "AIBTC/USD": {"active": True, "type": "spot", "quote": "USD"},
            "ETH/EUR": {"active": True, "type": "spot", "quote": "EUR"},
        }


def test_load_kraken_symbols_excludes_synthetic(caplog):
    ex = SyntheticExchange()
    with caplog.at_level(logging.INFO, logger=ml_logger.name):
        symbols = asyncio.run(load_kraken_symbols(ex))
    assert set(symbols) == {"BTC/USD", "ETH/EUR"}
    messages = [r.getMessage() for r in caplog.records if r.name == ml_logger.name]
    assert any("Excluded 1 synthetic/index pairs" in m for m in messages)


class DummyFetchExchange:
    def __init__(self):
        self.markets = {"BTC/USD": {"id": "BTC/USD"}}
        self.rateLimit = 0

    async def fetch_ohlcv(self, symbol, timeframe="1h", limit=100, **kwargs):
        return [[0, 0, 0, 0, 0, 0]]

    def market(self, symbol):
        return self.markets[symbol]


def test_load_ohlcv_skips_missing_symbol(caplog):
    ex = DummyFetchExchange()

    async def run():
        return await load_ohlcv(ex, "ETH/USD")

    with caplog.at_level(logging.DEBUG, logger=ml_logger.name):
        data = asyncio.run(run())
    assert data == []
    messages = [r.getMessage() for r in caplog.records if r.name == ml_logger.name]
    assert all("Unsupported symbol" not in m for m in messages)
