import pandas as pd


def detect_patterns(df: pd.DataFrame) -> set[str]:
    """Return a set of simple chart patterns detected in ``df``.

    The function inspects the latest candle and recent history for
    breakout and candlestick formations. Only a handful of patterns
    are recognized:

    ``"breakout"``  -- last close at a new high with a volume spike
    ``"breakdown"`` -- last close at a new low with a volume spike
    ``"hammer"``    -- small body with long lower shadow
    ``"shooting_star"`` -- small body with long upper shadow
    ``"doji"``      -- open and close nearly equal
    """
    patterns: set[str] = set()
    if df is None or len(df) < 2:
        return patterns

    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(last["close"] - last["open"])
    candle_range = last["high"] - last["low"]
    if candle_range == 0:
        return patterns
    upper = last["high"] - max(last["close"], last["open"])
    lower = min(last["close"], last["open"]) - last["low"]

    if body <= candle_range * 0.1:
        if upper > candle_range * 0.4 and lower <= candle_range * 0.1:
            patterns.add("shooting_star")
        if lower > candle_range * 0.4 and upper <= candle_range * 0.1:
            patterns.add("hammer")
        if upper > candle_range * 0.2 and lower > candle_range * 0.2:
            patterns.add("doji")

    lookback = min(len(df), 20)
    if len(df) >= 2:
        high_max = df["high"].rolling(lookback).max().iloc[-2]
        low_min = df["low"].rolling(lookback).min().iloc[-2]
        vol_mean = df["volume"].rolling(lookback).mean().iloc[-2]
    else:
        high_max = df["high"].max()
        low_min = df["low"].min()
        vol_mean = df["volume"].mean()

    if last["close"] >= high_max and last["volume"] > vol_mean * 1.5:
        patterns.add("breakout")
    if last["close"] <= low_min and last["volume"] > vol_mean * 1.5:
        patterns.add("breakdown")

    return patterns
