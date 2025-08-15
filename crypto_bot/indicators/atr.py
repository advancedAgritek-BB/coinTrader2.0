from __future__ import annotations

import pandas as pd

from crypto_bot.utils.indicator_cache import cache_series


def calc_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Return the latest Average True Range (ATR) value.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing ``high``, ``low`` and ``close`` columns.
    window : int, default 14
        Number of periods used for the ATR calculation.

    Returns
    -------
    float
        The most recent ATR value. ``0.0`` is returned when required
        columns are missing or the input is empty.
    """

    if df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return 0.0

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = tr.rolling(window, min_periods=window).mean()
    cached = cache_series(f"atr_{window}", df, atr_series, window)
    if cached.empty:
        return 0.0
    return float(cached.iloc[-1])


__all__ = ["calc_atr"]

