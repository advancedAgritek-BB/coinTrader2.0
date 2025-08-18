"""Utility functions for volatility and Average True Range (ATR)."""

from __future__ import annotations

import math
import pandas as pd
from ta.volatility import AverageTrueRange


def calc_atr(
    df: pd.DataFrame | None,
    window: int = 14,
    *,
    as_series: bool = True,
    **kwargs,
) -> pd.Series | float | None:
    """Compute the Average True Range using ``window`` or ``period``.

    Parameters
    ----------
    df : pandas.DataFrame | None
        Input OHLC data.
    window : int, default 14
        Window length for ATR calculation.
    as_series : bool, default True
        When ``True`` a :class:`pandas.Series` is returned, otherwise the
        latest ATR value is returned as ``float`` or ``None`` when
        insufficient data is provided.
    """

    period = kwargs.get("period")
    if period is not None:
        window = int(period)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    if df is None or df.empty or len(df) < max(2, int(window)):
        if as_series:
            if df is not None and "close" in df:
                return df["close"].iloc[:0]
            return pd.Series([], dtype=float)
        return None

    atr_indicator = AverageTrueRange(
        df["high"], df["low"], df["close"], window=int(window), fillna=False
    )
    atr_series = atr_indicator.average_true_range()
    if as_series:
        return atr_series
    if atr_series.empty:
        return None
    value = float(atr_series.iloc[-1])
    return 0.0 if math.isnan(value) else value

    # The following unreachable code is intentionally removed to avoid
    # ambiguous return paths and ensure a scalar is returned when requested.


def atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    """Return ATR as a percentage of the latest close price.

    The return value is a scalar percentage (0-100). ``nan`` is returned when
    ATR or price data are unavailable.
    """

    last_close = float(df["close"].iloc[-1])
    atr_val = calc_atr(df, period=period, as_series=False)
    if not (math.isfinite(atr_val) and last_close > 0):
        return float("nan")
    return 100.0 * atr_val / last_close


def normalize_score_by_volatility(
    df: pd.DataFrame, score: float, atr_period: int = 14
) -> float:
    """Normalize ``score`` by dividing by the latest ATR.

    If the ATR cannot be computed or is zero the ``score`` is returned
    unchanged. This helper is used to de-emphasise trading signals during
    periods of heightened volatility.
    """

    atr_val = calc_atr(df, period=atr_period, as_series=False)
    if atr_val is None or not math.isfinite(atr_val) or atr_val == 0:
        return float(score)
    return float(score) / float(atr_val)


__all__ = ["calc_atr", "atr_percent", "normalize_score_by_volatility"]

