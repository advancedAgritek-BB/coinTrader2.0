import numpy as np
import pandas as pd


def zscore(series: pd.Series, lookback: int = 250) -> pd.Series:
    """Return z-score relative to the last ``lookback`` observations."""
    if lookback <= 0 or len(series) < lookback:
        return pd.Series(dtype=float)
    window = series.tail(lookback)
    std = window.std()
    if std == 0 or pd.isna(std):
        return pd.Series(dtype=float)
    mean = window.mean()
    return (series - mean) / std


def last_window_zscore(series, window: int) -> float:
    """Z-score of the last point in the last ``window`` values of ``series``."""
    if series is None or len(series) < window:
        return np.nan
    w = series.iloc[-window:]
    std = w.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((w.iloc[-1] - w.mean()) / std)

