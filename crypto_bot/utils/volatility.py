from __future__ import annotations

import math
import pandas as pd

try:  # pragma: no cover - optional dependency
    from crypto_bot.indicators.atr import calc_atr  # type: ignore
except Exception:  # pragma: no cover - best effort
    calc_atr = None


def _fallback_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute ATR locally when the indicator import fails."""
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def atr_percent(df: pd.DataFrame, window: int = 14) -> float:
    """Return ATR as a percentage of the latest close price."""
    if df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return 0.0

    series = (
        calc_atr(df, period=window) if calc_atr is not None else _fallback_atr(df, window)
    )
    if series.empty:
        return 0.0

    atr = float(series.iloc[-1])
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

    The function compares the current ATR to a long-term average (default
    20-period). The score is multiplied by
    ``min(current_atr / long_term_avg_atr, 2.0)``. If ATR values are
    unavailable, the raw score is returned unchanged.
    """
    if raw_score == 0 or df.empty:
        return raw_score
    if not {"high", "low", "close"}.issubset(df.columns):
        return raw_score

    calc = calc_atr if calc_atr is not None else _fallback_atr
    current_series = calc(df, period=current_window)
    long_series = calc(df, period=long_term_window)
    if current_series.empty or long_series.empty:
        return raw_score

    current_atr = float(current_series.iloc[-1])
    long_term_atr = float(long_series.iloc[-1])
    if any(math.isnan(x) or x == 0 for x in (current_atr, long_term_atr)):
        return raw_score

    scale = min(current_atr / long_term_atr, 2.0)
    return raw_score * scale

