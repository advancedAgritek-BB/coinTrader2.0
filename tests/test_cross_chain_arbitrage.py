from crypto_bot.strategy import cross_chain_arbitrage
from crypto_bot import strategy_router

class DummyEx:
    def __init__(self, price):
        self.price = price
    def fetch_ticker(self, symbol):
        return {"last": self.price}

def test_cross_chain_arbitrage_long(monkeypatch):
    ex = DummyEx(99)
    monkeypatch.setattr(cross_chain_arbitrage, "fetch_solana_price", lambda s: 100)
    score, direction = cross_chain_arbitrage.generate_signal([ex], "BTC/USDT", threshold=0.005)
    assert direction == "long"
    assert score > 0

def test_router_cross_chain_arbitrage(monkeypatch):
    ex1 = DummyEx(110)
    ex2 = DummyEx(105)
    monkeypatch.setattr(cross_chain_arbitrage, "fetch_solana_price", lambda s: 100)
    cfg = {
        "cross_chain_arbitrage": {
            "enabled": True,
            "exchanges": [ex1, ex2],
            "symbol": "BTC/USDT",
            "threshold": 0.005,
        }
    }
    fn = strategy_router.route("trending", "cex", cfg)
    score, direction = fn(None)
    assert direction == "short"
    assert score > 0
