import pandas as pd

from crypto_bot.strategy import trend_bot


def _df_trend(volume_last: float, high_equals_close: bool = False):
    close = pd.Series(range(1, 61), dtype=float)
    if high_equals_close:
        high = close
    else:
        high = close + 0.5
    low = close - 0.5
    volume = pd.Series([100.0] * 59 + [volume_last])
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


def test_no_signal_when_volume_below_ma():
    df = _df_trend(50.0)
    score, direction = trend_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


def test_long_signal_with_filters():
    df = _df_trend(150.0)
    score, direction = trend_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0.0


def test_donchian_confirmation_blocks_false_breakout():
    df = _df_trend(150.0)
    cfg = {"donchian_confirmation": True}
    score, direction = trend_bot.generate_signal(df, cfg)
    assert direction == "none"
    assert score == 0.0


def test_donchian_confirmation_allows_breakout():
    df = _df_trend(150.0, high_equals_close=True)
    cfg = {"donchian_confirmation": True}
    score, direction = trend_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0.0
