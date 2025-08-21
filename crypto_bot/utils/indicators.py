"""Lightweight technical indicators used across the codebase."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_series(x) -> pd.Series:
    """Return ``x`` as a ``pandas.Series`` of ``float64``."""

    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        return pd.Series(x, dtype="float64")
    raise TypeError(f"Unsupported input type: {type(x)}")


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with a given ``period``."""

    s = _to_series(series).astype(float)
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Welles Wilder's True Range."""

    high_s, low_s, close_s = map(_to_series, (high, low, close))
    prev_close = close_s.shift(1)
    tr1 = high_s - low_s
    tr2 = (high_s - prev_close).abs()
    tr3 = (low_s - prev_close).abs()
    h, low_series, c = map(_to_series, (high, low, close))
    prev_close = c.shift(1)
    tr1 = h - low_series
    tr2 = (h - prev_close).abs()
    tr3 = (low_series - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range using Wilder's smoothing."""

    tr = true_range(high, low, close)
    # Wilder's smoothing uses an ``alpha`` of ``1/period``
    tr = true_range(high, low, close)
    # Wilder's smoothing
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calc_atr(
    df: pd.DataFrame,
    window: int = 14,
    *,
    period: int | None = None,
    high: str = "high",
    low: str = "low",
    close: str = "close",
) -> pd.Series:
    """Convenience wrapper around :func:`atr` for OHLC data frames.

    Parameters
    ----------
    df : pandas.DataFrame
        Input OHLC data containing columns for ``high``, ``low`` and ``close``
        prices.  Column names can be customised via ``high``, ``low`` and
        ``close`` parameters.
    window : int, default 14
        Lookback window for the ATR calculation.
    period : int, optional
        Deprecated alias for ``window`` kept for backwards compatibility. When
        provided it takes precedence over ``window``.
    high, low, close : str
        Column names for the respective OHLC values.
    """

    if period is not None:
        window = int(period)

    cols = {high, low, close}
    if not cols.issubset(df.columns):
        msg = f"calc_atr expects columns {cols}, got {list(df.columns)}"
        raise ValueError(msg)

    h = df[high].astype(float)
    l = df[low].astype(float)
    c = df[close].astype(float)
    tr = true_range(h, l, c)
    return tr.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""

    c = _to_series(close).astype(float)
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    gain = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


__all__ = ["ema", "true_range", "atr", "calc_atr", "rsi"]
