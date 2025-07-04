import pandas as pd
from crypto_bot.strategy import breakout_bot


def _make_df(prices, volumes):
    return pd.DataFrame({
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": volumes,
    })


def test_long_breakout_with_consolidation():
    prices = [100 + (1 if i % 2 == 0 else -1) for i in range(35)] + [100] * 6 + [104]
    volumes = [100] * 41 + [300]
    df = _make_df(prices, volumes)
    cfg = {"consolidation_period": 3, "contraction_threshold": 1.0}
    score, direction = breakout_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0


def test_short_breakout_with_consolidation():
    prices = [100 + (1 if i % 2 == 0 else -1) for i in range(35)] + [100] * 6 + [96]
    volumes = [100] * 41 + [300]
    df = _make_df(prices, volumes)
    cfg = {"consolidation_period": 3, "contraction_threshold": 1.0}
    score, direction = breakout_bot.generate_signal(df, cfg)
    assert direction == "short"
    assert score > 0


def test_requires_contraction():
    prices = [100 + (1 if i % 2 == 0 else -1) for i in range(40)] + [104]
    volumes = [100] * 40 + [300]
    df = _make_df(prices, volumes)
    cfg = {"consolidation_period": 3, "contraction_threshold": 1.0}
    score, direction = breakout_bot.generate_signal(df, cfg)
    assert direction == "none"
    assert score == 0.0
