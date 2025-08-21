import importlib
import numpy as np
import pandas as pd
import pytest

stat_arb_bot = importlib.import_module("crypto_bot.strategy.stat_arb_bot")


@pytest.fixture
def correlated_dfs():
    rs = np.random.RandomState(0)
    base = np.cumsum(rs.normal(0, 1, 50)) + 100
    noise = rs.normal(0, 0.1, 50)
    df_a = pd.DataFrame({"close": base})
    df_b = pd.DataFrame({"close": base + noise})
    return df_a, df_b


@pytest.fixture
def not_cointegrated_dfs():
    rs = np.random.RandomState(1)
    a = np.cumsum(rs.normal(0, 1, 50))
    b = np.cumsum(rs.normal(0, 1.5, 50))
    df_a = pd.DataFrame({"close": a})
    df_b = pd.DataFrame({"close": b})
    return df_a, df_b


def test_signal_direction_and_score(correlated_dfs):
    df_a, df_b = correlated_dfs
    df_a.loc[df_a.index[-1], "close"] += 5
    score, direction = stat_arb_bot.generate_signal(
        df_a,
        df_b,
        config={"zscore_threshold": 1.0, "lookback": 20},
    )
    assert score > 0
    assert direction == "short"


def test_no_signal_when_not_cointegrated(not_cointegrated_dfs):
    df_a, df_b = not_cointegrated_dfs
    score, direction = stat_arb_bot.generate_signal(df_a, df_b, config={"zscore_threshold": 1.0})
    assert score == 0.0
    assert direction == "none"
