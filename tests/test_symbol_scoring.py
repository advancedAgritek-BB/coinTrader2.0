import asyncio
import pytest

from crypto_bot.utils.symbol_scoring import score_symbol

class DummyEx:
    def __init__(self):
        self.markets = {"BTC/USD": {"created": 0}}
    async def fetch_ticker(self, symbol):
        return {}
    def milliseconds(self):
        return 0

@pytest.mark.asyncio
async def test_score_symbol_zero_weights_error():
    cfg = {"symbol_score_weights": {"volume": 0, "change": 0, "spread": 0, "age": 0, "latency": 0}}
    with pytest.raises(ValueError):
        await score_symbol(DummyEx(), "BTC/USD", 1000, 1.0, 0.1, cfg)
