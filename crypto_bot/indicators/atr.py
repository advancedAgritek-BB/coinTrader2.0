from __future__ import annotations

import pandas as pd


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return the Average True Range for ``df``.

    Parameters
    ----------
    df: pandas.DataFrame
        Must contain ``high``, ``low`` and ``close`` columns.
    period: int, default 14
        Rolling window size.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()
    """
    Simple ATR (SMA of True Range). Columns required: 'high', 'low', 'close'.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window=period, min_periods=period).mean()

import pandas as pd

from crypto_bot.utils.indicator_cache import cache_series


def calc_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Calculate the Average True Range using cached values."""
    lookback = window
    recent = df.iloc[-(lookback + 1) :]
    high_low = recent["high"] - recent["low"]
    high_close = (recent["high"] - recent["close"].shift()).abs()
    low_close = (recent["low"] - recent["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = tr.rolling(window).mean()
    cached = cache_series(f"atr_{window}", df, atr_series, lookback)
    return float(cached.iloc[-1])
