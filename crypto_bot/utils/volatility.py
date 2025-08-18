import logging
import math
import pandas as pd
from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)


def calc_atr(
    df: pd.DataFrame | None, window: int = 14, **kwargs
) -> pd.Series:
    """Compute the Average True Range using ``window`` or ``period``."""

    period = kwargs.get("period")
    if period is not None:
        window = int(period)

    if df is None or df.empty or len(df) < max(2, int(window)):
        if df is not None and "close" in df:
            return df["close"].iloc[:0]
        return pd.Series([], dtype=float)

    atr_indicator = AverageTrueRange(
        df["high"], df["low"], df["close"], window=int(window), fillna=False
    )
    return atr_indicator.average_true_range()


def atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    """Return the latest ATR value as a fraction of the close price."""

    if df is None or df.empty or "close" not in df:
        return float("nan")

    atr_series = calc_atr(df, period=period)
    if getattr(atr_series, "empty", False):
        return float("nan")

    price = float(df["close"].iloc[-1])
    if price == 0 or math.isnan(price):
        return float("nan")

    atr_value = float(atr_series.iloc[-1]) if hasattr(atr_series, "iloc") else float(
        atr_series
    )
    return atr_value / price


def normalize_score_by_volatility(
    df: pd.DataFrame,
    score: float,
    atr_period: int = 14,
    eps: float = 1e-8,
) -> float:
    """Scale ``score`` by the latest ATR percentage.

    If ATR cannot be computed, returns ``score`` unchanged.
    """
    try:
        atr = calc_atr(df, period=atr_period)
        if getattr(atr, "empty", False) or df is None or df.empty or "close" not in df:
            return score

        last_close = df["close"].iloc[-1]
        if last_close == 0 or math.isnan(last_close):
            return score

        last_atr = atr.iloc[-1] if hasattr(atr, "iloc") else float(atr)
        vol = last_atr / last_close

        if vol < 0.1 and score / max(vol, eps) > score:
            return score * vol
        return score / max(vol, eps) if vol else score
    except Exception:
        logger.exception(
            "normalize_score_by_volatility: ATR unavailable; returning unnormalized score",
        )
        return score


__all__ = ["calc_atr", "atr_percent", "normalize_score_by_volatility"]
