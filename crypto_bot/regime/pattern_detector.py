import pandas as pd


def detect_patterns(df: pd.DataFrame) -> dict[str, float]:
    """Return confidence scores for simple chart patterns detected in ``df``.

    The function inspects the latest candle and recent history for
    breakout and candlestick formations. Only a handful of patterns
    are recognized.  The return value maps pattern names to a confidence
    between 0 and 1.  Patterns with zero confidence are omitted.

    ``"breakout"``  -- last close at a new high with a volume spike
    ``"breakdown"`` -- last close at a new low with a volume spike
    ``"hammer"``    -- small body with long lower shadow
    ``"shooting_star"`` -- small body with long upper shadow
    ``"doji"``      -- open and close nearly equal
    """
    scores: dict[str, float] = {}
    if df is None or len(df) < 2:
        return scores

    last = df.iloc[-1]

    body = abs(last["close"] - last["open"])
    candle_range = last["high"] - last["low"]
    if candle_range == 0:
        return scores
    upper = last["high"] - max(last["close"], last["open"])
    lower = min(last["close"], last["open"]) - last["low"]

    if body <= candle_range * 0.1:
        if upper > candle_range * 0.4 and lower <= candle_range * 0.1:
            scores["shooting_star"] = 1.0
        if lower > candle_range * 0.4 and upper <= candle_range * 0.1:
            scores["hammer"] = 1.0
        if upper > candle_range * 0.2 and lower > candle_range * 0.2:
            scores["doji"] = 1.0

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
        scores["breakout"] = 1.0
    if last["close"] <= low_min and last["volume"] > vol_mean * 1.5:
        scores["breakdown"] = 1.0

    return scores
