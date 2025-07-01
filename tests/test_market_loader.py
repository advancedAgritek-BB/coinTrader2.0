from crypto_bot.utils.market_loader import load_kraken_symbols

class DummyExchange:
    def load_markets(self):
        return {
            "BTC/USD": {"active": True},
            "ETH/USD": {"active": True},
            "XRP/USD": {"active": False},
        }

def test_load_kraken_symbols_returns_active():
    ex = DummyExchange()
    symbols = load_kraken_symbols(ex)
    assert set(symbols) == {"BTC/USD", "ETH/USD"}

def test_excluded_symbols_are_removed():
    ex = DummyExchange()
    symbols = load_kraken_symbols(ex, exclude=["ETH/USD"])
    assert set(symbols) == {"BTC/USD"}
