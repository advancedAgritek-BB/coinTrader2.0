from __future__ import annotations

"""Utilities for confirming multi-timeframe trend strength."""

from typing import Optional
import pandas as pd
import ta


def _strong_trend(df: pd.DataFrame) -> bool:
    """Return ``True`` for a bullish trend with above average ADX."""
    if df is None or df.empty or len(df) < 50:
        return False
    df = df.copy()
    df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)
    adx = ta.trend.adx(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx
    adx_avg = adx.rolling(20).mean()
    latest = df.iloc[-1]
    return bool(latest["ema20"] > latest["ema50"] and latest["adx"] >= max(20, adx_avg.iloc[-1]))


def confirm_multi_tf_trend(low_df: pd.DataFrame, high_df: Optional[pd.DataFrame]) -> bool:
    """Check if ``low_df`` and ``high_df`` both exhibit strong bullish trend."""
    if high_df is None or high_df.empty:
        return False
    return _strong_trend(low_df) and _strong_trend(high_df)
