from __future__ import annotations

import logging
import math
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)


def calc_atr(
    df: pd.DataFrame | None,
    window: int = 14,
    **kwargs,
) -> pd.Series | float:
    """
    Backward-compatible ATR that accepts ``window`` or ``period``.

    If ``df`` is missing or does not contain enough rows, an empty ``Series``
    aligned to ``df`` (when possible) or ``0.0`` is returned.
    """
    if "period" in kwargs and kwargs["period"] is not None:
        window = kwargs["period"]

    if df is None or df.empty or len(df) < max(2, int(window)):
        if df is not None and "close" in df:
            return df["close"].iloc[:0]
        return 0.0

    return AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=int(window),
        fillna=False,
    ).average_true_range()


def atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    """Return the most recent ATR value as a fraction of the latest close."""
    atr = calc_atr(df, period=period)
    if getattr(atr, "empty", False) or df is None or len(df) == 0:
        return np.nan

    last_close = df["close"].iloc[-1]
    last_atr = atr.iloc[-1] if hasattr(atr, "iloc") else float(atr)
    return (last_atr / last_close) if last_close else np.nan


def normalize_score_by_volatility(
    df: pd.DataFrame,
    score: float,
    atr_period: int = 14,
    eps: float = 1e-8,
) -> float:
    """Scale ``score`` by the latest ATR percentage.

    If ATR cannot be computed, returns ``score`` unchanged.
    """
    try:
        atr = calc_atr(df, period=atr_period)
        if getattr(atr, "empty", False) or len(df) == 0:
            return score

        last_close = df["close"].iloc[-1]
        last_atr = atr.iloc[-1] if hasattr(atr, "iloc") else float(atr)
        vol = (last_atr / last_close) if last_close else 0.0
        return score / max(vol, eps) if vol else score
    except Exception:
        logger.exception(
            "normalize_score_by_volatility: ATR unavailable; returning unnormalized score",
        )
        return score


__all__ = ["atr_percent", "normalize_score_by_volatility", "calc_atr"]

