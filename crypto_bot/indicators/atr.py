from __future__ import annotations

import pandas as pd

from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.logger import indicator_logger


def calc_atr(df: pd.DataFrame, window: int = 14, **kwargs) -> float:
    """Return the latest Average True Range (ATR) value.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing ``high``, ``low`` and ``close`` columns.
    window : int, default 14
        Number of periods used for the ATR calculation.
    period : int, optional
        Deprecated alias for ``window`` kept for backwards compatibility.

    Returns
    -------
    float
        The most recent ATR value. ``0.0`` is returned when required
        columns are missing or the input is empty.
    """

    period = kwargs.get("period")
    if period is not None and window == 14:
        try:
            window = int(period)
        except (TypeError, ValueError):
            pass

    if df.empty or not {"high", "low", "close"}.issubset(df.columns):
        indicator_logger.warning(
            "ATR calculation skipped due to missing columns or empty data"
        )
        return 0.0

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = tr.rolling(window, min_periods=window).mean()
    cached = cache_series(f"atr_{window}", df, atr_series, window)
    if cached.empty:
        indicator_logger.warning("ATR cache miss for period %d", window)
        return 0.0
    value = float(cached.iloc[-1])
    indicator_logger.info("ATR(%d) computed %.6f", window, value)
    return value


__all__ = ["calc_atr"]

