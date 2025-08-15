import math  # for math.isnan checks

import pandas as pd
from crypto_bot.indicators.atr import calc_atr


def atr_percent(df: pd.DataFrame, window: int = 14) -> float:
    """Return ATR as a percentage of the latest close price."""
    if df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return 0.0
    result = calc_atr(df, window)
    if isinstance(result, pd.Series):
        if result.empty:
            return 0.0
        atr = float(result.iloc[-1])
    else:
        atr = float(result)
    price = float(df["close"].iloc[-1])
    if price == 0 or math.isnan(atr) or math.isnan(price):
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

    cur_res = calc_atr(df, current_window)
    long_res = calc_atr(df, long_term_window)
    current_atr = float(cur_res.iloc[-1] if isinstance(cur_res, pd.Series) else cur_res)
    long_term_atr = float(long_res.iloc[-1] if isinstance(long_res, pd.Series) else long_res)
    if math.isnan(current_atr) or math.isnan(long_term_atr) or long_term_atr == 0:
        return raw_score

    scale = min(current_atr / long_term_atr, 2.0)
    return raw_score * scale
