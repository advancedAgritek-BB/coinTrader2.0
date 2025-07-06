import numpy as np
import pandas as pd

from crypto_bot.regime.pattern_detector import detect_patterns


def _base_df(rows: int = 30) -> pd.DataFrame:
    close = np.linspace(1, 2, rows)
    high = close + 0.1
    low = close - 0.1
    volume = np.arange(rows) + 100
    return pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_detect_patterns_breakout():
    df = _base_df()
    df.loc[df.index[-1], "close"] = df["high"].max() + 0.5
    df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "close"] + 0.1
    df.loc[df.index[-1], "low"] = df.loc[df.index[-1], "close"] - 0.1
    df.loc[df.index[-1], "volume"] = df["volume"].mean() * 2

    patterns = detect_patterns(df)
    assert isinstance(patterns, dict)
    assert patterns.get("breakout", 0) > 0


def test_detect_patterns_hammer():
    df = _base_df()
    df.loc[df.index[-1], "open"] = 2.0
    df.loc[df.index[-1], "close"] = 2.05
    df.loc[df.index[-1], "high"] = 2.1
    df.loc[df.index[-1], "low"] = 1.8

    patterns = detect_patterns(df)
    assert isinstance(patterns, dict)
    assert patterns.get("hammer", 0) > 0
