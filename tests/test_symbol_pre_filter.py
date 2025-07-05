import asyncio
import json
from crypto_bot.utils.symbol_pre_filter import filter_symbols

CONFIG = {"symbol_filter": {"min_volume_usd": 50000}}

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
                "v": ["600", "600"],
                "p": ["100", "100"],
                "o": "99",
            },
            "XXBTZUSD": {
                "a": ["51", "1", "1"],
                "b": ["50", "1", "1"],
                "c": ["51", "1"],
                "v": ["400", "400"],
                "p": ["100", "100"],
                "o": "51",
            },
        }
    }


def test_filter_symbols(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )
    symbols = asyncio.run(filter_symbols(DummyExchange(), ["ETH/USD", "BTC/USD"], CONFIG))
    assert symbols == ["ETH/USD"]


class DummyExchangeList:
    # markets_by_id values may be lists of market dictionaries
    markets_by_id = {"XETHZUSD": [{"symbol": "ETH/USD"}]}


def test_filter_symbols_handles_list_values(monkeypatch):
    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )
    symbols = asyncio.run(filter_symbols(DummyExchangeList(), ["ETH/USD"], CONFIG))
    assert symbols == ["ETH/USD"]
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
    assert symbols == ["ETH/USD"]


def test_non_dict_market_entry(monkeypatch):
    class BadExchange:
        markets_by_id = {"XETHZUSD": ["ETH/USD"]}

    monkeypatch.setattr(
        "crypto_bot.utils.symbol_pre_filter._fetch_ticker_async", fake_fetch
    )
    symbols = asyncio.run(filter_symbols(BadExchange(), ["ETH/USD"], CONFIG))
    assert symbols == ["XETHZUSD"]


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
    assert symbols == ["ETH/USD"]
