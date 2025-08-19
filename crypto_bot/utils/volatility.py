"""Utility functions for volatility and Average True Range (ATR)."""

from __future__ import annotations

import math
import pandas as pd
from ta.volatility import AverageTrueRange


def calc_atr(
    df: pd.DataFrame | None,
    length: int | None = None,
    *,
    window: int | None = None,
    period: int | None = None,
    **_,
) -> pd.Series:
    """Compute the Average True Range.

    ``length`` is the preferred parameter name. ``window`` and ``period`` are
    accepted for backward compatibility and mapped to ``length``. A
    :class:`pandas.Series` of ATR values is always returned.
    """

    if length is None:
        if window is not None:
            length = window
        elif period is not None:
            length = period
    if length is None:
        length = 14

    if df is None or df.empty or len(df) < max(2, int(length)):
        if df is not None and "close" in df:
            return df["close"].iloc[:0]
        return pd.Series([], dtype=float)

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    atr_indicator = AverageTrueRange(
        high, low, close, window=int(length), fillna=False
    )
    return atr_indicator.average_true_range()


def atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    """Return ATR as a percentage of the latest close price.

    The return value is a scalar percentage (0-100). ``nan`` is returned when
    ATR or price data are unavailable.
    """

    last_close = float(df["close"].iloc[-1])
    atr_series = calc_atr(df, period=period)
    atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else float("nan")
    if not (math.isfinite(atr_val) and last_close > 0):
        return float("nan")
    return 100.0 * atr_val / last_close


def normalize_score_by_volatility(
    df: pd.DataFrame, score: float, atr_period: int = 14
) -> float:
    """Normalize ``score`` by dividing by the latest ATR.

    If the ATR cannot be computed or is zero the ``score`` is returned
    unchanged. This helper is used to de-emphasise trading signals during
    periods of heightened volatility.
    """

    atr_series = calc_atr(df, period=atr_period)
    if atr_series.empty:
        return float(score)
    atr_val = float(atr_series.iloc[-1])
    if not math.isfinite(atr_val) or atr_val == 0:
        return float(score)
    return float(score) / float(atr_val)


__all__ = ["calc_atr", "atr_percent", "normalize_score_by_volatility"]

