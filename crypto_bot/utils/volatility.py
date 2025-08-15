import math
import pandas as pd
import ta


def _atr(df: pd.DataFrame, window: int) -> float:
    """Return the latest ATR value for ``df`` using TA library.

    A lightweight helper to avoid importing :mod:`crypto_bot.volatility_filter`,
    which would otherwise create a circular dependency during module import.
    """
    if df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return 0.0

    series = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=window
    )
    if series.empty:
        return 0.0

    value = float(series.iloc[-1])
    return 0.0 if math.isnan(value) else value


def atr_percent(df: pd.DataFrame, window: int = 14) -> float:
    """Return ATR as a percentage of the latest close price."""
    atr = _atr(df, window)
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

    The function compares the current ATR to a long‑term average (default
    20‑period). The score is multiplied by
    ``min(current_atr / long_term_avg_atr, 2.0)``. If ATR values are
    unavailable, the raw score is returned unchanged.
    """
    if raw_score == 0 or df.empty:
        return raw_score
    if not {"high", "low", "close"}.issubset(df.columns):
        return raw_score

    current_atr = _atr(df, window=current_window)
    long_term_atr = _atr(df, window=long_term_window)
    if any(
        math.isnan(x) or x == 0 for x in [current_atr, long_term_atr]
    ):
        return raw_score

    scale = min(current_atr / long_term_atr, 2.0)
    return raw_score * scale
