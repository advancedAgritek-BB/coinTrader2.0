from __future__ import annotations

import math
import pandas as pd

from crypto_bot.indicators.atr import calc_atr


def atr_percent(df: pd.DataFrame, window: int = 14) -> float:
    """Return ATR as a percentage of the latest close price."""

    atr = calc_atr(df, window)
    if atr == 0:
        return 0.0

    price = float(df["close"].iloc[-1])
    if price == 0 or math.isnan(price):
        return 0.0
    return atr / price * 100


def normalize_score_by_volatility(
    df: pd.DataFrame,
    raw_score: float,
    current_window: int = 5,
    long_term_window: int = 20,
) -> float:
    """Scale ``raw_score`` based on market volatility.

    The score is multiplied by ``min(current_atr / long_term_atr, 2.0)``. If
    ATR values are unavailable the original score is returned.
    """

    if raw_score == 0 or df.empty:
        return raw_score
    if not {"high", "low", "close"}.issubset(df.columns):
        return raw_score

    current_atr = calc_atr(df, current_window)
    long_term_atr = calc_atr(df, long_term_window)
    if any(math.isnan(x) or x == 0 for x in (current_atr, long_term_atr)):
        return raw_score
"""Compatibility layer for volatility helpers.

This module re-exports functions from :mod:`crypto_bot.volatility` so existing
imports continue to work after the helpers were centralized.
"""

from __future__ import annotations

import sys

from crypto_bot import volatility as _vol

# Re-export the base module so attribute patches affect the shared implementation.
sys.modules[__name__] = _vol


__all__ = ["atr_percent", "normalize_score_by_volatility", "calc_atr"]

