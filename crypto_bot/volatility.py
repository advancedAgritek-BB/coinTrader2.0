import math
import pandas as pd
import ta

try:  # pragma: no cover - optional dependency
    from crypto_bot.indicators.atr import calc_atr  # type: ignore
except Exception:  # pragma: no cover - best effort
    calc_atr = None


def _fallback_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Compute ATR locally when the indicator import fails."""
    return ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=window
    )


def _atr(df: pd.DataFrame, window: int) -> float:
    """Return the latest ATR value for ``df``."""
    if df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return 0.0

    result = (
        calc_atr(df, window) if calc_atr is not None else _fallback_atr(df, window)
    )
    if isinstance(result, pd.Series):
        if result.empty:
            return 0.0
        value = float(result.iloc[-1])
    else:
        value = float(result)

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

    current_atr = _atr(df, current_window)
    long_term_atr = _atr(df, long_term_window)
    if any(math.isnan(x) or x == 0 for x in (current_atr, long_term_atr)):
        return raw_score

    scale = min(current_atr / long_term_atr, 2.0)
    return raw_score * scale


__all__ = ["_atr", "atr_percent", "normalize_score_by_volatility", "calc_atr"]
