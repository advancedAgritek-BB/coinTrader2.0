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

@pytest.mark.asyncio
async def test_score_symbol_dataframe_normalization():
    import pandas as pd
    from crypto_bot.utils import symbol_scoring as sc

    df = pd.DataFrame({
        "open": [100, 101, 102],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
        "close": [100.5, 101.5, 102.5],
        "volume": [100, 100, 100],
    })

    volume_usd = float(df["volume"].sum() * df["close"].iloc[-1])
    change_pct = float((df["close"].iloc[-1] - df["open"].iloc[0]) / df["open"].iloc[0] * 100)
    spread_pct = float((df["high"].max() - df["low"].min()) / df["close"].iloc[-1] * 100)
    liquidity = 0.75

    cfg = {"symbol_score_weights": {"age": 0.0, "latency": 0.0}}
    score = await sc.score_symbol(DummyEx(), "BTC/USD", volume_usd, change_pct, spread_pct, liquidity, cfg)

    weights = sc.DEFAULT_WEIGHTS.copy()
    weights.update(cfg["symbol_score_weights"])
    total = sum(weights.values())
    raw_score = (
        volume_usd * weights["volume"]
        + abs(change_pct) * weights["change"]
        + (100 - spread_pct) * weights["spread"]
        + liquidity * weights["liquidity"]
    ) / total

    assert score != pytest.approx(raw_score)
