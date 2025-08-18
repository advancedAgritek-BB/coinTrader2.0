"""Utility helpers for dealing with volatility values."""
from __future__ import annotations

import math
import logging
import numpy as np
import pandas as pd

from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)


def calc_atr(df: pd.DataFrame, window: int = 14, **kwargs) -> pd.Series:
    """Compute the Average True Range using ``window`` or ``period``."""

    period = kwargs.get("period")
    if period is not None:
        window = int(period)

    if df is None or df.empty or len(df) < max(2, int(window)):
        return (
            df["close"].iloc[:0] if df is not None and "close" in df else pd.Series([], dtype=float)
        )

    atr_indicator = AverageTrueRange(
        df["high"], df["low"], df["close"], window=int(window), fillna=False
    )
    return atr_indicator.average_true_range()


def atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    """Return the latest ATR value as a fraction of the close price."""

    if df is None or df.empty or "close" not in df:
        return float(np.nan)

    atr_series = calc_atr(df, period=period)
    if getattr(atr_series, "empty", False):
        return float(np.nan)

    price = float(df["close"].iloc[-1])
    if price == 0 or math.isnan(price):
        return float(np.nan)

    atr_value = float(atr_series.iloc[-1]) if hasattr(atr_series, "iloc") else float(atr_series)
    return atr_value / price


def normalize_score_by_volatility(df, score, atr_period: int = 14):
    """Scale ``score`` by current ATR percentage of price.

    If ATR isn't available, returns ``score`` unchanged.
    """

    try:
        atr_series = calc_atr(df, period=atr_period)
        if getattr(atr_series, "empty", False) or df is None or df.empty:
            return score
        last_close = df["close"].iloc[-1]
        last_atr = atr_series.iloc[-1] if hasattr(atr_series, "iloc") else float(atr_series)
        vol = (last_atr / last_close) if last_close else 0.0
        return score / max(vol, 1e-9) if vol else score
    except Exception:
        logger.exception(
            "normalize_score_by_volatility: ATR unavailable; returning unnormalized score"
        )
        return score


__all__ = ["atr_percent", "normalize_score_by_volatility", "calc_atr"]
