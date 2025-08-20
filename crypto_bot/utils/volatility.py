"""Utility functions for volatility and Average True Range (ATR)."""
from __future__ import annotations

import math
import pandas as pd
from ta.volatility import AverageTrueRange


def calc_atr(df: pd.DataFrame | None, period: int | None = None, window: int = 14) -> pd.Series:
    """Return the Average True Range (ATR) as a :class:`pandas.Series`.

    ``period`` takes precedence over ``window`` when both are provided. The
    underlying calculation uses :class:`ta.volatility.AverageTrueRange`.
    """

    n = int(period or window or 14)
    if df is None or df.empty or len(df) < max(2, n):
        if df is not None and "close" in df:
            return df["close"].iloc[:0]
        return pd.Series([], dtype=float)

    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=n)
    return pd.Series(atr.average_true_range())


def atr_percent(df: pd.DataFrame, period: int | None = None, window: int = 14) -> float:
    """Return ATR as a percentage of the latest close price."""

    last_close = float(df["close"].iloc[-1])
    atr_series = calc_atr(df, period=period, window=window)
    if atr_series.empty:
        return float("nan")
    atr_val = float(atr_series.iloc[-1])
    if not (math.isfinite(atr_val) and last_close > 0):
        return float("nan")
    return 100.0 * atr_val / last_close


def normalize_score_by_volatility(df: pd.DataFrame, score: float, atr_period: int = 14) -> float:
    """Normalize ``score`` by dividing by the latest ATR.

    If the ATR cannot be computed or is non‑positive the ``score`` is returned
    unchanged. This helper is used to de‑emphasise trading signals during
    periods of heightened volatility.
    """

    atr = calc_atr(df, period=atr_period)
    if hasattr(atr, "iloc"):
        if len(atr) == 0:
            return float(score)
        atr_last = float(atr.iloc[-1])
    else:
        atr_last = float(atr)
    if atr_last > 0:
        score /= atr_last
    return float(score)


__all__ = ["calc_atr", "atr_percent", "normalize_score_by_volatility"]

