import pandas as pd

from crypto_bot.strategy import micro_scalp_bot


def _df(prices, volumes):
    return pd.DataFrame({"close": prices, "volume": volumes})


def test_micro_scalp_long_signal():
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = _df(prices, volumes)
    score, direction = micro_scalp_bot.generate_signal(df)
    assert direction == "long"
    assert 0 < score <= 1


def test_volume_filter_blocks_signal():
    prices = list(range(1, 11))
    volumes = [1] * 10
    df = _df(prices, volumes)
    cfg = {"micro_scalp": {"volume_threshold": 2, "volume_window": 5}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert direction == "none"
    assert score == 0.0


def test_trend_filter_blocks_long_signal():
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = _df(prices, volumes)

    higher_prices = list(range(20, 9, -1))
    higher_df = _df(higher_prices, [100] * len(higher_prices))
    cfg = {"micro_scalp": {"trend_fast": 3, "trend_slow": 5}}

    score, direction = micro_scalp_bot.generate_signal(df, cfg, higher_df=higher_df)
    assert (score, direction) == (0.0, "none")


def test_trend_filter_allows_long_signal():
    prices = list(range(1, 11))
    volumes = [100] * 10
    df = _df(prices, volumes)

    higher_prices = list(range(10, 21))
    higher_df = _df(higher_prices, [100] * len(higher_prices))
    cfg = {"micro_scalp": {"trend_fast": 3, "trend_slow": 5}}

    score, direction = micro_scalp_bot.generate_signal(df, cfg, higher_df=higher_df)
    assert direction == "long"
    assert score > 0
