import pandas as pd

from crypto_bot.strategy import breakout_bot


def _make_df(prices, volumes):
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": volumes,
        }
    )


def test_long_breakout_signal():
    prices = [100] * 25 + [103]
    volumes = [100] * 25 + [300]
    df = _make_df(prices, volumes)
    score, direction, atr = breakout_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0


def test_short_breakout_signal():
    prices = [100] * 25 + [97]
    volumes = [100] * 25 + [300]
    df = _make_df(prices, volumes)
    score, direction, atr = breakout_bot.generate_signal(df)
    assert direction == "short"
    assert score > 0


def test_requires_squeeze():
    prices = list(range(80, 106))
    volumes = [100] * 26
    df = _make_df(prices, volumes)
    score, direction, atr = breakout_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0
