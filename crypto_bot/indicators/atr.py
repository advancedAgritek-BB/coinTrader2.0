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
