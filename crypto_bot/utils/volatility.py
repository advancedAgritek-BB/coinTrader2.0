"""Utility helpers for dealing with volatility values."""
from __future__ import annotations

import math
import logging
import pandas as pd

from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)


def calc_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute the Average True Range for ``df`` using a given ``window``."""

    if len(df) < window:
        return pd.Series([0] * len(df))
    atr_indicator = AverageTrueRange(
        df["high"], df["low"], df["close"], window=window, fillna=False
    )
    return atr_indicator.average_true_range()


def atr_percent(df: pd.DataFrame, window: int = 14) -> float:
    """Return the latest ATR value as a percentage of the close price."""

    if df.empty or "close" not in df:
        return 0.0
    atr_series = calc_atr(df, window=window)
    if atr_series.empty:
        return 0.0
    atr_value = float(atr_series.iloc[-1])
    price = float(df["close"].iloc[-1])
    if price == 0 or math.isnan(price):
        return 0.0
    return (atr_value / price) * 100.0


def normalize_score_by_volatility(df, score, atr_period: int = 14, eps: float = 1e-8):
    """Scales ``score`` by current ATR.

    If ATR isn't available, returns ``score`` unchanged.
    """

    try:
        atr = calc_atr(df, window=atr_period)
        current_atr = float(atr.iloc[-1]) if hasattr(atr, "iloc") else float(atr)
        denom = max(abs(current_atr), eps)
        return score / denom
    except Exception:
        logger.exception(
            "normalize_score_by_volatility: ATR unavailable; returning unnormalized score"
        )
        return score


__all__ = ["atr_percent", "normalize_score_by_volatility", "calc_atr"]
