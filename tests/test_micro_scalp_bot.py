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
