import json
from crypto_bot.utils.symbol_pre_filter import filter_symbols

class DummyExchange:
    markets_by_id = {
        "XETHZUSD": {"symbol": "ETH/USD"},
        "XXBTZUSD": {"symbol": "BTC/USD"},
    }


def fake_get(url, timeout=10):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
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

    return Resp()


def test_filter_symbols(monkeypatch):
    monkeypatch.setattr("crypto_bot.utils.symbol_pre_filter.requests.get", fake_get)
    symbols = filter_symbols(DummyExchange(), ["ETH/USD", "BTC/USD"])
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
    monkeypatch.setattr("crypto_bot.utils.symbol_pre_filter.requests.get", fake_get)
    symbols = filter_symbols(ex, ["ETH/USD"])
    assert ex.called is True
    assert symbols == ["ETH/USD"]


def test_non_dict_market_entry(monkeypatch):
    class BadExchange:
        markets_by_id = {"XETHZUSD": ["ETH/USD"]}

    monkeypatch.setattr("crypto_bot.utils.symbol_pre_filter.requests.get", fake_get)
    symbols = filter_symbols(BadExchange(), ["ETH/USD"])
    assert symbols == ["XETHZUSD"]
