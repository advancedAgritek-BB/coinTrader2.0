import asyncio
import pytest

pytestmark = pytest.mark.asyncio

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
    cfg = {"symbol_score_weights": {"volume": 0, "change": 0, "spread": 0, "age": 0, "latency": 0, "liquidity": 0}}
    with pytest.raises(ValueError):
        await score_symbol(DummyEx(), "BTC/USD", 1000, 1.0, 0.1, 1.0, cfg)


@pytest.mark.asyncio
async def test_liquidity_weight():
    cfg = {"symbol_score_weights": {"volume": 0, "change": 0, "spread": 0, "age": 0, "latency": 0, "liquidity": 1.0}}
    score = await score_symbol(DummyEx(), "BTC/USD", 0, 0, 0, 0.5, cfg)
    assert score == pytest.approx(0.5)
