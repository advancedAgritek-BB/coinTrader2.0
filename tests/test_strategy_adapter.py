import pandas as pd
import pytest

from crypto_bot import strategies


@pytest.fixture(autouse=True)
def _provider():
    def provider(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        return pd.DataFrame({"close": [1, 2, 3]})

    strategies.set_ohlcv_provider(provider)
    yield


@pytest.mark.asyncio
async def test_native_score_preferred():
    class Strat:
        async def score(self, symbols, timeframes):
            assert symbols == ["BTC/USD"]
            assert timeframes == ["1m"]
            return {("BTC/USD", "1m"): 1}

    res = await strategies.score(Strat(), symbols=["BTC/USD"], timeframes=["1m"])
    assert res == {("BTC/USD", "1m"): 1}


@pytest.mark.asyncio
async def test_generate_signal_uses_dataframe():
    calls = {}

    class Strat:
        def generate_signal(self, df, symbol, timeframe):
            calls["df"] = df
            return "sig"

    res = await strategies.score(Strat(), symbols=["ETH/USD"], timeframes=["5m"])
    assert ("ETH/USD", "5m") in res and res[("ETH/USD", "5m")] == "sig"
    assert isinstance(calls["df"], pd.DataFrame)


@pytest.mark.asyncio
async def test_pair_strategy_support():
    class PairStrat:
        PAIRS = [("BTC/USD", "ETH/USD")]

        def generate_signal(self, df_a, df_b, symbol_a=None, symbol_b=None, timeframe=None):
            assert isinstance(df_a, pd.DataFrame)
            assert isinstance(df_b, pd.DataFrame)
            return "pair"

    res = await strategies.score(
        PairStrat(), symbols=["BTC/USD", "ETH/USD"], timeframes=["1m"]
    )
    assert res[("BTC/USD|ETH/USD", "1m")] == "pair"
