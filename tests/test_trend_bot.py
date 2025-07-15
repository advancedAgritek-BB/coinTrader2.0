import pandas as pd

from crypto_bot.strategy import trend_bot
import numpy as np
import ta


def _df_trend(volume_last: float, high_equals_close: bool = False):
    close = pd.Series(range(1, 61), dtype=float)
    if high_equals_close:
        high = close
    else:
        high = close + 0.5
    low = close - 0.5
    volume = pd.Series([100.0] * 59 + [volume_last])
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


def _df_adx_range():
    np.random.seed(0)
    slope = 0.05
    close = pd.Series(np.linspace(100, 100 + slope * 59, 60) + np.random.normal(0, 0.2, 60))
    high = close + 0.2
    low = close - 0.2
    volume = pd.Series([200.0] * 59 + [220.0])
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


def test_no_signal_when_volume_below_ma():
    df = _df_trend(50.0)
    score, direction = trend_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


def test_long_signal_with_filters():
    df = _df_trend(150.0)
    cfg = {"donchian_confirmation": False}
    score, direction = trend_bot.generate_signal(df, cfg)
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


def test_rsi_zscore(monkeypatch):
    df = _df_trend(150.0)
    monkeypatch.setattr(
        trend_bot.stats,
        "zscore",
        lambda s, lookback=3: pd.Series([2] * len(s), index=s.index),
    )
    cfg = {"indicator_lookback": 3, "rsi_overbought_pct": 90, "donchian_confirmation": False}
    score, direction = trend_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0


def test_adx_threshold(monkeypatch):
    df = _df_adx_range()
    adx_val = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=7).adx().iloc[-1]
    assert 25 <= adx_val <= 30

    monkeypatch.setattr(
        trend_bot.stats,
        "zscore",
        lambda s, lookback=250: pd.Series([2] * len(s), index=s.index),
    )

    score, direction = trend_bot.generate_signal(
        df, {"donchian_confirmation": False, "adx_threshold": adx_val + 1}
    )
    assert direction == "none"
    assert score == 0.0

    score, direction = trend_bot.generate_signal(
        df, {"donchian_confirmation": False, "adx_threshold": adx_val - 1}
    )
    assert direction != "none"
    assert score > 0
