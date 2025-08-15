"""Utility helpers for dealing with volatility values."""

from __future__ import annotations

import math
import pandas as pd

from crypto_bot.utils.indicators import calc_atr


def normalize_score_by_volatility(
    score: float | pd.DataFrame,
    df: pd.DataFrame | float,
    atr_period: int = 14,
    floor: float = 0.25,
    ceil: float = 2.0,
) -> float:
    """Scale ``score`` by the recent volatility of ``df``.

    ``atr_period`` determines the look‑back window for the ATR calculation. The
    resulting ATR percentage is mapped to the range ``[floor, ceil]`` and the
    ``score`` is multiplied by this factor. When insufficient data is supplied
    the ``floor`` multiplier is applied.
    """

    if isinstance(score, pd.DataFrame):
        df, score = score, float(df)
        current_window, long_term_window = 5, 20
        current_atr = calc_atr(df, current_window)
        long_term_atr = calc_atr(df, long_term_window)
        if any(math.isnan(x) or x == 0 for x in (current_atr, long_term_atr)):
            return score
        scale = min(current_atr / long_term_atr, 2.0)
        return score * scale

    if len(df) < atr_period + 1:
        return floor * score

    atr_pct = (calc_atr(df, period=atr_period) / df["close"]).iloc[-1]
    low, high = 0.001, 0.03  # 0.1% .. 3% daily-ish
    x = max(min(float(atr_pct), high), low)
    k = (x - low) / (high - low)
    factor = floor + k * (ceil - floor)
    return score * factor


__all__ = ["normalize_score_by_volatility", "calc_atr"]
