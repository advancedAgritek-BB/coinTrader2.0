"""Compatibility wrappers for volatility helpers.

This module exposes :func:`calc_atr`, :func:`atr_percent` and
:func:`normalize_score_by_volatility` while allowing tests to monkeypatch
``calc_atr`` directly on this module.
"""

from __future__ import annotations

import math
import pandas as pd

from crypto_bot.volatility import atr_percent as _atr_percent, calc_atr as _base_calc_atr

# ``calc_atr`` is defined at module scope so tests can monkeypatch it.
calc_atr = _base_calc_atr


def atr_percent(df: pd.DataFrame, window: int = 14) -> float:
    """Proxy for :func:`crypto_bot.volatility.atr_percent`."""
    return _atr_percent(df, window)


def normalize_score_by_volatility(
    df: pd.DataFrame,
    raw_score: float,
    current_window: int = 5,
    long_term_window: int = 20,
) -> float:
    """Scale ``raw_score`` based on market volatility.

    This mirrors :func:`crypto_bot.volatility.normalize_score_by_volatility` but
    references ``calc_atr`` from this module so it can be patched in tests.
    """

    if raw_score == 0 or df.empty:
        return raw_score
    if not {"high", "low", "close"}.issubset(df.columns):
        return raw_score

    current_atr = calc_atr(df, current_window)
    long_term_atr = calc_atr(df, long_term_window)
    if any(math.isnan(x) or x == 0 for x in (current_atr, long_term_atr)):
        return raw_score

    scale = min(current_atr / long_term_atr, 2.0)
    return raw_score * scale


__all__ = ["atr_percent", "normalize_score_by_volatility", "calc_atr"]

