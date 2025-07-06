import pandas as pd


def detect_patterns(df: pd.DataFrame) -> dict[str, float]:
    """Return a mapping of detected chart patterns to their strength.

    The function inspects the latest candle and recent history for
    breakout and candlestick formations. Only a handful of patterns
    are recognized:

    ``"breakout"``  -- last close at a new high with a volume spike
    ``"breakdown"`` -- last close at a new low with a volume spike
    ``"hammer"``    -- small body with long lower shadow
    ``"shooting_star"`` -- small body with long upper shadow
    ``"doji"``      -- open and close nearly equal
    """
    patterns: dict[str, float] = {}
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
            patterns["shooting_star"] = upper / candle_range
        if lower > candle_range * 0.4 and upper <= candle_range * 0.1:
            patterns["hammer"] = lower / candle_range
        if upper > candle_range * 0.2 and lower > candle_range * 0.2:
            patterns["doji"] = 1 - body / candle_range

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
        strength = (last["close"] - high_max) / high_max
        patterns["breakout"] = max(strength, 0.0) + 1.0
    if last["close"] <= low_min and last["volume"] > vol_mean * 1.5:
        strength = (low_min - last["close"]) / low_min
        patterns["breakdown"] = max(strength, 0.0) + 1.0

    # Detect simple ascending triangle
    lookback = min(len(df), 5)
    recent = df.iloc[-lookback:]
    highs = recent["high"]
    lows = recent["low"]
    if (
        highs.max() - highs.min() <= highs.mean() * 0.005
        and lows.diff().dropna().gt(0).all()
    ):
        patterns["ascending_triangle"] = 1.5

    return patterns
