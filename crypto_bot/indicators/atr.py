from __future__ import annotations

import pandas as pd

from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.logger import indicator_logger


def calc_atr(
    df: pd.DataFrame,
    window: int = 14,
    *,
    period: int | None = None,
    high: str = "high",
    low: str = "low",
    close: str = "close",
) -> float:
    """Return the latest Average True Range (ATR) value.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing columns for ``high``, ``low`` and ``close`` prices.
        Column names can be customised via ``high``, ``low`` and ``close``.
    window : int, default 14
        Number of periods used for the ATR calculation.
    period : int, optional
        Deprecated alias for ``window`` kept for backwards compatibility. When
        provided it takes precedence over ``window``.
    high, low, close : str
        Column names for the respective OHLC values.

    Returns
    -------
    float
        The most recent ATR value. ``0.0`` is returned when required columns are
        missing or the input is empty.
    """

    if period is not None:
        try:
            window = int(period)
        except (TypeError, ValueError):
            pass

    cols = {high, low, close}
    if df.empty or not cols.issubset(df.columns):
        indicator_logger.warning(
            "ATR calculation skipped due to missing columns or empty data",
        )
        return 0.0

    hi = df[high].astype(float)
    lo = df[low].astype(float)
    cl = df[close].astype(float)
    tr = pd.concat([
        (hi - lo).abs(),
        (hi - cl.shift()).abs(),
        (lo - cl.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(window, min_periods=window).mean()
    cached = cache_series(f"atr_{window}", df, atr_series, window)
    if cached.empty:
        indicator_logger.warning("ATR cache miss for period %d", window)
        return 0.0
    value = float(cached.iloc[-1])
    indicator_logger.info("ATR(%d) computed %.6f", window, value)
    return value


__all__ = ["calc_atr"]

