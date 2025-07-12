import pandas as pd

from crypto_bot.strategy import mean_bot


def _df_with_drop(price: float) -> pd.DataFrame:
    base = [100.0] * 19 + [price]
    data = {
        "open": base,
        "high": [p + 1 for p in base],
        "low": [p - 1 for p in base],
        "close": base,
        "volume": [100] * len(base),
    }
    return pd.DataFrame(data)


def test_long_signal_on_big_drop():
    df = _df_with_drop(80.0)
    score, direction = mean_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0


def test_short_signal_on_big_spike():
    df = _df_with_drop(120.0)
    score, direction = mean_bot.generate_signal(df)
    assert direction == "short"
    assert score > 0


def test_skewed_rsi_quantile(monkeypatch):
    df = _df_with_drop(100.0)

    monkeypatch.setattr(
        mean_bot.ta.momentum,
        "rsi",
        lambda s, window=14: pd.Series([30] * len(s), index=s.index),
    )
    monkeypatch.setattr(
        mean_bot.stats,
        "zscore",
        lambda s, lookback=250: pd.Series([2] * 19 + [-1], index=s.index),
    )

    score, direction = mean_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0


def test_empty_rsi_z_fallback(monkeypatch):
    df = _df_with_drop(80.0)

    monkeypatch.setattr(
        mean_bot.stats, "zscore", lambda s, lookback=250: pd.Series(dtype=float)
    )

    score, direction = mean_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0


