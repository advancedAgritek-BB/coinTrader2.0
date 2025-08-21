"""Utility functions for volatility and Average True Range (ATR)."""
from __future__ import annotations

import math
import pandas as pd

from crypto_bot.utils.indicators import calc_atr as _calc_atr


def calc_atr(
    df: pd.DataFrame | None,
    window: int = 14,
    *,
    period: int | None = None,
    high: str = "high",
    low: str = "low",
    close: str = "close",
) -> pd.Series:
    """Return the Average True Range (ATR) as a :class:`pandas.Series`.

    ``period`` is accepted as an alias for ``window`` to maintain backwards
    compatibility.  Column names for ``high``, ``low`` and ``close`` may also be
    customised.
    """

    n = int(period or window or 14)
    if df is None or df.empty or len(df) < max(2, n):
        if df is not None and close in df:
            return df[close].iloc[:0]
        return pd.Series([], dtype=float)

    return _calc_atr(df, window=n, high=high, low=low, close=close)


def atr_percent(
    df: pd.DataFrame,
    period: int | None = None,
    window: int = 14,
    *,
    high: str = "high",
    low: str = "low",
    close: str = "close",
) -> float:
    """Return ATR as a percentage of the latest close price."""

    last_close = float(df[close].iloc[-1])
    kwargs = {}
    if high != "high":
        kwargs["high"] = high
    if low != "low":
        kwargs["low"] = low
    if close != "close":
        kwargs["close"] = close
    atr_series = calc_atr(df, period=period, window=window, **kwargs)
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
    high: str = "high",
    low: str = "low",
    close: str = "close",
) -> float:
    """Normalize ``score`` by dividing by the latest ATR.

    If the ATR cannot be computed or is non‑positive the ``score`` is returned
    unchanged. This helper is used to de‑emphasise trading signals during
    periods of heightened volatility.
    """

    kwargs = {}
    if high != "high":
        kwargs["high"] = high
    if low != "low":
        kwargs["low"] = low
    if close != "close":
        kwargs["close"] = close
    atr = calc_atr(df, period=atr_period, **kwargs)
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

