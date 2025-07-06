import pandas as pd
import numpy as np

from crypto_bot.regime.pattern_detector import detect_patterns
from crypto_bot.regime.regime_classifier import classify_regime_with_patterns


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
    assert patterns.get("breakout", 0) == 1.0
    assert "breakout" in patterns
    assert patterns["breakout"] > 0
    assert isinstance(patterns["breakout"], float)


def test_classify_regime_includes_patterns():
    df = _base_df()
    df.loc[df.index[-1], "close"] = df["high"].max() + 0.5
    df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "close"] + 0.1
    df.loc[df.index[-1], "low"] = df.loc[df.index[-1], "close"] - 0.1
    df.loc[df.index[-1], "volume"] = df["volume"].mean() * 2

    regime, patterns = classify_regime_with_patterns(df)
    regime, patterns = classify_regime(df)
    assert regime == "sideways"
    assert patterns.get("breakout", 0) == 1.0
    assert regime == "breakout"
    assert isinstance(patterns, dict)
    assert patterns.get("breakout", 0) > 0


def test_detect_patterns_bullish_engulfing():
    df = _base_df()
    # previous candle bearish, small
    df.loc[df.index[-2], "open"] = 2.0
    df.loc[df.index[-2], "close"] = 1.9
    # current candle bullish and engulfs previous body
    df.loc[df.index[-1], "open"] = 1.85
    df.loc[df.index[-1], "close"] = 2.1
    df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "close"] + 0.1
    df.loc[df.index[-1], "low"] = df.loc[df.index[-1], "open"] - 0.1

    patterns = detect_patterns(df)
    assert isinstance(patterns, dict)
    assert patterns.get("bullish_engulfing", 0) > 0
    assert "breakout" in patterns
    assert patterns["breakout"] > 0
