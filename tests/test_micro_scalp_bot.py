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


def test_min_momentum_blocks_signal():
    prices = [10 + i * 0.01 for i in range(10)]
    volumes = [100] * 10
    df = _df(prices, volumes)
    cfg = {"micro_scalp": {"min_momentum_pct": 0.01}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_confirm_bars_blocks_fresh_cross():
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    volumes = [100] * len(prices)
    df = _df(prices, volumes)
    cfg = {"micro_scalp": {"confirm_bars": 2}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")


def test_fresh_cross_only_signal():
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    volumes = [100] * len(prices)
    df = _df(prices, volumes)
    cfg = {"micro_scalp": {"fresh_cross_only": True, "confirm_bars": 1}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert direction == "short"
    assert score > 0


def test_fresh_cross_only_requires_change():
    prices = [1, 2, 3, 4, 5, 6, 7, 8, 3, 2, 1]
    volumes = [100] * len(prices)
    df = _df(prices, volumes)
    cfg = {"micro_scalp": {"fresh_cross_only": True}}
    score, direction = micro_scalp_bot.generate_signal(df, cfg)
    assert (score, direction) == (0.0, "none")
