from __future__ import annotations

import pandas as pd

from crypto_bot.utils.indicators import calc_atr as _calc_atr


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    atr = _calc_atr(df, period=period)
    return (atr / df["close"]).fillna(0.0)


def too_flat(
    df: pd.DataFrame,
    atr_period: int = 14,
    threshold: float = 0.004,
) -> bool:
    """
    Heuristic: return True if ATR% (median of the last ``atr_period``) is below
    ``threshold``. ``threshold`` is ATR divided by close (e.g., ``0.004`` =
    ``0.4%``).
    """
    if len(df) < max(atr_period, 20):
        return True
    ap = atr_pct(df, period=atr_period).iloc[-atr_period:].median()
    return float(ap) < threshold


# Keep legacy import path working for existing callers
def calc_atr(df: pd.DataFrame, period: int = 14):
    return _calc_atr(df, period=period)
