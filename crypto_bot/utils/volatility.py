"""Utility functions for volatility and Average True Range (ATR)."""
from __future__ import annotations

import math
import pandas as pd


def calc_atr(
    df: pd.DataFrame | None,
    length: int = 14,
    *,
    period: int | None = None,
    window: int | None = None,
) -> pd.Series:
    """Compute the Average True Range.

    The lookback window may be provided via ``period``, ``window`` or
    ``length``. These names are treated as aliases with the first
    non-``None`` value taking precedence. A :class:`pandas.Series` of ATR
    values is returned regardless of the input.
    """

    n = int(period or window or length or 14)
    if df is None or df.empty or len(df) < max(2, n):
        if df is not None and "close" in df:
            return df["close"].iloc[:0]
        return pd.Series([], dtype=float)

    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def atr_percent(
    df: pd.DataFrame,
    length: int = 14,
    *,
    period: int | None = None,
    window: int | None = None,
) -> float:
    """Return ATR as a percentage of the latest close price."""

    last_close = float(df["close"].iloc[-1])
    atr_series = calc_atr(df, length=length, period=period, window=window)
    if atr_series.empty:
        return float("nan")
    atr_val = float(atr_series.iloc[-1])
    if not (math.isfinite(atr_val) and last_close > 0):
        return float("nan")
    return 100.0 * atr_val / last_close


def normalize_score_by_volatility(
    df: pd.DataFrame,
    score: float,
    atr_period: int = 14,
    *,
    period: int | None = None,
    length: int | None = None,
    window: int | None = None,
) -> float:
    """Normalize ``score`` by dividing by the latest ATR.

    If the ATR cannot be computed or is zero the ``score`` is returned
    unchanged. This helper is used to de‑emphasise trading signals during
    periods of heightened volatility.
    """

    n = int(period or length or window or atr_period)
    atr_series = calc_atr(df, period=n)
    if atr_series.empty:
        return float(score)
    atr_val = float(atr_series.iloc[-1])
    if not math.isfinite(atr_val) or atr_val == 0:
        return float(score)
    return float(score) / float(atr_val)


__all__ = ["calc_atr", "atr_percent", "normalize_score_by_volatility"]

