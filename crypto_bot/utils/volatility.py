"""Utility helpers for dealing with volatility values."""
from __future__ import annotations

import math
import inspect
import logging
import pandas as pd

from crypto_bot.indicators import calc_atr

logger = logging.getLogger(__name__)


def _calc_atr_compat(df, atr_period):
    """Call ``calc_atr(df, ...)`` using whatever parameter name it supports."""
    try:
        params = set(inspect.signature(calc_atr).parameters.keys())
    except Exception:
        params = set()
    for pname in ("period", "window", "length", "n"):
        if pname in params:
            return calc_atr(df, **{pname: atr_period})
    try:
        # As a last resort, try positional
        return calc_atr(df, atr_period)
    except TypeError:
        # Or no-arg beyond df
        return calc_atr(df)


def atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    """Return ATR as percentage of the latest close price."""

    if df.empty or "close" not in df:
        return 0.0
    price = float(df["close"].iloc[-1])
    if price == 0 or math.isnan(price):
        return 0.0
    atr = calc_atr(df, period=period)
    return 0.0 if math.isnan(atr) else float(atr / price * 100.0)


def normalize_score_by_volatility(df, score, atr_period=14, eps=1e-8):
    """Scales ``score`` by current ATR.

    If ATR isn't available, returns ``score`` unchanged.
    """

    try:
        atr = _calc_atr_compat(df, atr_period)
        current_atr = float(atr.iloc[-1]) if hasattr(atr, "iloc") else float(atr)
        denom = max(abs(current_atr), eps)
        return score / denom
    except Exception:
        logger.exception(
            "normalize_score_by_volatility: ATR unavailable; returning unnormalized score"
        )
        return score


__all__ = ["atr_percent", "normalize_score_by_volatility", "calc_atr"]
