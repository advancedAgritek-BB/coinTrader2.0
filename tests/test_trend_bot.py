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
    df = _df_trend(150.0, high_equals_close=True)
    score, direction = trend_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0.0


def test_donchian_confirmation_blocks_false_breakout():
    df = _df_trend(150.0)
    score, direction = trend_bot.generate_signal(df)
    assert direction == "none"
    assert score == 0.0


def test_donchian_confirmation_allows_breakout():
    df = _df_trend(150.0, high_equals_close=True)
    score, direction = trend_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0.0


def test_disable_donchian_allows_breakout():
    df = _df_trend(150.0)
    cfg = {"donchian_confirmation": False}
    score, direction = trend_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0.0


def test_rsi_zscore(monkeypatch):
    df = _df_trend(150.0, high_equals_close=True)
    monkeypatch.setattr(
        trend_bot.stats,
        "zscore",
        lambda s, lookback=3: pd.Series([0] * (len(s) - 1) + [2], index=s.index),
    )
    cfg = {"indicator_lookback": 3, "rsi_overbought_pct": 90}
    score, direction = trend_bot.generate_signal(df, cfg)
    assert direction == "long"
    assert score > 0


def test_reversal_cross_signal(monkeypatch):
    df = _df_trend(150.0, high_equals_close=True)

    def fake_ema(s, window=5):
        val = 1 if window == 5 else 2
        arr = [val] * (len(s) - 1) + [2 if window == 5 else 1]
        return pd.Series(arr, index=s.index)

    class FakeADX:
        def __init__(self, *args, **kwargs):
            self.len = len(args[0])

        def adx(self):
            return pd.Series([30] * self.len)

    monkeypatch.setattr(trend_bot.ta.trend, "ema_indicator", fake_ema)
    monkeypatch.setattr(trend_bot.ta.momentum, "rsi", lambda s, window=14: pd.Series([20] * len(s), index=s.index))
    monkeypatch.setattr(trend_bot.stats, "zscore", lambda s, lookback=3: pd.Series([float("nan")] * len(s), index=s.index))
    monkeypatch.setattr(trend_bot.ta.trend, "ADXIndicator", FakeADX)

    score, direction = trend_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0
