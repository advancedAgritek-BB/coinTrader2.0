import pandas as pd
import pytest

from crypto_bot.strategy import micro_scalp_bot


@pytest.fixture
def make_df():
    def _factory(prices, volumes, opens=None, highs=None, lows=None):
        opens = prices if opens is None else opens
        highs = [p + 0.5 for p in prices] if highs is None else highs
        lows = [p - 0.5 for p in prices] if lows is None else lows
        return pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            }
        )

    return _factory


def test_micro_scalp_long_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert 0 < score <= 1


def test_cross_with_momentum_and_wick(make_df):
    prices = [10, 9, 8, 7, 6, 5, 6, 7, 8, 9]
    volumes = [100] * len(prices)
    cfg = {
        "micro_scalp": {
            "ema_fast": 3,
            "ema_slow": 8,
            "lower_wick_pct": 0.3,
            "min_momentum_pct": 0.02,
            "fresh_cross_only": False,
        }
    }
    df = make_df(prices, volumes)
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0


def test_volume_filter_blocks_signal(make_df):
    prices = list(range(1, 11))
    volumes = [1] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"min_vol_z": 2, "volume_window": 5, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_atr_filter_blocks_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"atr_period": 3, "min_atr_pct": 0.2, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_trend_filter_blocks_long_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)

    higher_prices = list(range(20, 9, -1))
    higher_df = make_df(higher_prices, [100] * len(higher_prices))
    cfg = {"micro_scalp": {"trend_fast": 3, "trend_slow": 5, "fresh_cross_only": False}}

    score, direction = micro_scalp_bot.generate_signal(df, cfg, higher_df=higher_df)
    assert (score, direction) == (0.0, "none")


def test_trend_filter_allows_long_signal(make_df):
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = make_df(prices, volumes)

    higher_prices = list(range(10, 21))
    higher_df = make_df(higher_prices, [100] * len(higher_prices))
    cfg = {"micro_scalp": {"trend_fast": 3, "trend_slow": 5, "fresh_cross_only": False}}

    score, direction = micro_scalp_bot.generate_signal(df, cfg, higher_df=higher_df)
    assert direction == "long"
    assert score > 0


def test_min_momentum_blocks_signal(make_df):
    prices = [10 + i * 0.01 for i in range(10)]
    volumes = [100] * 10
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"min_momentum_pct": 0.01, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_confirm_bars_blocks_fresh_cross(make_df):
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"confirm_bars": 2, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_fresh_cross_only_signal(make_df):
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"fresh_cross_only": True, "confirm_bars": 1}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert direction == "short"
    assert score > 0


def test_fresh_cross_only_requires_change(make_df):
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 3, 2, 1]
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    cfg = {"micro_scalp": {"fresh_cross_only": True}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_wick_filter_blocks_long(make_df):
    prices = list(range(1, 11))
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    df.loc[df.index[-1], "low"] = df["close"].iloc[-1] - 0.05
    cfg = {"micro_scalp": {"wick_pct": 0.2, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_wick_filter_blocks_short(make_df):
    prices = list(range(10, 0, -1))
    volumes = [100] * len(prices)
    df = make_df(prices, volumes)
    df.loc[df.index[-1], "high"] = df["close"].iloc[-1] + 0.05
    cfg = {"micro_scalp": {"wick_pct": 0.2, "fresh_cross_only": False}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


@pytest.mark.parametrize(
    "prices,volumes,cfg",
    [
        (
            [10 + i * 0.01 for i in range(10)],
            [100] * 10,
            {"micro_scalp": {"min_momentum_pct": 0.01, "fresh_cross_only": False}},
        ),
        (
            list(range(1, 11)),
            [1] * 10,
            {"micro_scalp": {"min_vol_z": 2, "volume_window": 5, "fresh_cross_only": False}},
        ),
        (
            list(range(1, 11)),
            [100] * 10,
            {"micro_scalp": {"atr_period": 3, "min_atr_pct": 0.2, "fresh_cross_only": False}},
        ),
    ],
)
def test_filters_return_none(make_df, prices, volumes, cfg):
    df = make_df(prices, volumes)
    assert micro_scalp_bot.generate_signal(df, cfg) == (0.0, "none")
