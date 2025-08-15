from __future__ import annotations

import numpy as np
import pandas as pd


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        return pd.Series(x, dtype="float64")
    raise TypeError(f"Unsupported input type: {type(x)}")


def ema(series: pd.Series, period: int) -> pd.Series:
    s = _to_series(series).astype(float)
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h, low_series, c = map(_to_series, (high, low, close))
    prev_close = c.shift(1)
    tr1 = h - low_series
    tr2 = (h - prev_close).abs()
    tr3 = (low_series - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr = true_range(high, low, close)
    # Wilder's smoothing
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    cols = {"high", "low", "close"}
    if not cols.issubset(df.columns):
        message = "calc_atr expects columns {}, got {}".format(
            cols,
            list(df.columns),
        )
        raise ValueError(message)
    return atr(df["high"], df["low"], df["close"], period=period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    c = _to_series(close).astype(float)
    delta = c.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    gain = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))
