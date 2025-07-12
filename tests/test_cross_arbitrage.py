import types
from crypto_bot.strategy import cross_arbitrage
from crypto_bot import strategy_router

class DummyEx:
    def __init__(self, price):
        self.price = price
    def fetch_ticker(self, symbol):
        return {"last": self.price}

def test_cross_arbitrage_long():
    ex_a = DummyEx(99)
    ex_b = DummyEx(100)
    score, direction = cross_arbitrage.generate_signal(ex_a, ex_b, "BTC/USDT", threshold=0.005)
    assert direction == "long"
    assert score > 0

def test_router_cross_arbitrage(monkeypatch):
    ex_a = DummyEx(110)
    ex_b = DummyEx(100)
    cfg = {"cross_arbitrage": {"enabled": True, "exchange_a": ex_a, "exchange_b": ex_b, "symbol": "BTC/USDT", "threshold": 0.005}}
    fn = strategy_router.route("trending", "cex", cfg)
    score, direction = fn(None)
    assert direction == "short"
    assert score > 0
