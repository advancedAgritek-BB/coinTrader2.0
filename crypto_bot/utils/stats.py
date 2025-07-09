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
