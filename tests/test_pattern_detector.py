import numpy as np
import pandas as pd

from crypto_bot.regime.pattern_detector import detect_patterns
from crypto_bot.regime.regime_classifier import (
    classify_regime,
    classify_regime_with_patterns,
)


def _base_df(rows: int = 30) -> pd.DataFrame:
    close = np.linspace(1, 2, rows)
    high = close + 0.1
    low = close - 0.1
    volume = np.arange(rows) + 100
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_detect_patterns_breakout():
    df = _base_df()
    df.loc[df.index[-1], "close"] = df["high"].max() + 0.5
    df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "close"] + 0.1
    df.loc[df.index[-1], "low"] = df.loc[df.index[-1], "close"] - 0.1
    df.loc[df.index[-1], "volume"] = df["volume"].mean() * 2

    patterns = detect_patterns(df)
    assert isinstance(patterns, dict)
    assert patterns.get("breakout", 0) > 0
    assert "breakout" in patterns
    assert patterns["breakout"] > 0
    assert isinstance(patterns["breakout"], float)


def test_detect_patterns_hammer():
    df = _base_df()
    df.loc[df.index[-1], "open"] = 2.0
    df.loc[df.index[-1], "close"] = 2.05
    df.loc[df.index[-1], "high"] = 2.1
    df.loc[df.index[-1], "low"] = 1.8
    df.loc[df.index[-1], "close"] = df["high"].max() + 0.5
    df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "close"] + 0.1
    df.loc[df.index[-1], "low"] = df.loc[df.index[-1], "close"] - 0.1
    df.loc[df.index[-1], "volume"] = df["volume"].mean() * 2

    patterns = detect_patterns(df)
    assert "breakout" in patterns
    assert patterns["breakout"] > 0


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
    assert "bullish_engulfing" in patterns


def test_head_and_shoulders_detection():
    close = [1, 1.2, 1.0, 1.5, 1.0, 1.2, 1.0]
    df = pd.DataFrame(
        {
            "open": close,
            "high": np.array(close) + 0.05,
            "low": np.array(close) - 0.05,
            "close": close,
            "volume": np.arange(len(close)) + 100,
        }
    )
    patterns = detect_patterns(df)
    assert "head_and_shoulders" in patterns


def test_pattern_min_conf_threshold():
    df = _base_df()
    df.loc[df.index[-1], "close"] = df["high"].max() + 0.5
    df.loc[df.index[-1], "high"] = df.loc[df.index[-1], "close"] + 0.1
    df.loc[df.index[-1], "low"] = df.loc[df.index[-1], "close"] - 0.1
    df.loc[df.index[-1], "volume"] = df["volume"].mean() * 2

    patterns = detect_patterns(df, min_conf=1.1)
    assert "breakout" not in patterns


def test_hammer_trend_boost():
    def _hammer_df(trend: str) -> pd.DataFrame:
        rows = 30
        if trend == "down":
            close = np.linspace(2, 1, rows)
        else:
            close = np.linspace(1, 2, rows)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": np.arange(rows) + 100,
            }
        )
        df.loc[df.index[-1], "open"] = df.loc[df.index[-1], "close"] + 0.05
        df.loc[df.index[-1], "low"] = df.loc[df.index[-1], "close"] - 0.3
        return df

    down_score = detect_patterns(_hammer_df("down")).get("hammer", 0)
    up_score = detect_patterns(_hammer_df("up")).get("hammer", 0)
    assert down_score > up_score
