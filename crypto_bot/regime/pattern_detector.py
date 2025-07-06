import pandas as pd


def detect_patterns(df: pd.DataFrame) -> dict[str, float]:
    """Return confidence scores for simple chart patterns detected in ``df``.

    The latest candle and recent history are inspected for a small
    selection of candlestick formations and breakout signals. Only
    patterns that are detected with a confidence between 0 and 1 are
    returned.
    """

    """Return a mapping of detected chart patterns to their strength.

    The latest candles are scanned for classic candlestick formations
    and simple breakout signals.  Each pattern is assigned a confidence
    value between ``0`` and ``1`` and only those with non–zero confidence
    are returned.

    ``"breakout"``         -- last close at a new high with volume spike
    ``"breakdown"``        -- last close at a new low with volume spike
    ``"hammer"``           -- small body with long lower shadow
    ``"shooting_star"``    -- small body with long upper shadow
    ``"doji"``             -- open and close nearly equal
    ``"bullish_engulfing"``  -- last candle engulfs previous bearish body
    ``"bearish_engulfing"`` -- last candle engulfs previous bullish body
    ``"inside_bar"``       -- current range is inside previous bar
    ``"three_bar_reversal"`` -- two bars one way then strong reversal
    ``"volume_spike"``     -- volume significantly above average
    """
    scores: dict[str, float] = {}
    patterns: dict[str, float] = {}
    if df is None or len(df) < 2:
        return patterns

    prev = df.iloc[-2]
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(last["close"] - last["open"])
    candle_range = last["high"] - last["low"]
    if candle_range == 0:
        return patterns

    upper = last["high"] - max(last["close"], last["open"])
    lower = min(last["close"], last["open"]) - last["low"]

    body_ratio = body / candle_range
    upper_ratio = upper / candle_range
    lower_ratio = lower / candle_range

    if body_ratio <= 0.4:
        hammer_score = max(0.0, (lower_ratio - upper_ratio) * (1 - body_ratio))
        if hammer_score > 0:
            patterns["hammer"] = min(hammer_score, 1.0)
        shooting_score = max(0.0, (upper_ratio - lower_ratio) * (1 - body_ratio))
        if shooting_score > 0:
            patterns["shooting_star"] = min(shooting_score, 1.0)

    doji_score = max(0.0, 1.0 - body_ratio / 0.1)
    if doji_score > 0:
        patterns["doji"] = min(doji_score, 1.0)

    lookback = min(len(df), 20)
    high_max = df["high"].rolling(lookback).max().iloc[-2]
    low_min = df["low"].rolling(lookback).min().iloc[-2]
    vol_mean = df["volume"].rolling(lookback).mean().iloc[-2]

    body_prev = abs(prev["close"] - prev["open"])
    body_last = abs(last["close"] - last["open"])
    if (
        last["close"] > last["open"]
        and prev["close"] < prev["open"]
        and last["open"] <= prev["close"]
        and last["close"] >= prev["open"]
    ):
        patterns["bullish_engulfing"] = min(1.0, body_last / (body_prev + 1e-9))

    if (
        last["close"] < last["open"]
        and prev["close"] > prev["open"]
        and last["open"] >= prev["close"]
        and last["close"] <= prev["open"]
    ):
        patterns["bearish_engulfing"] = min(1.0, body_last / (body_prev + 1e-9))

    if last["high"] < prev["high"] and last["low"] > prev["low"]:
        range_prev = prev["high"] - prev["low"]
        inside_score = 1.0 - (
            prev["high"] - last["high"] + last["low"] - prev["low"]
        ) / (range_prev + 1e-9)
        if inside_score > 0:
            patterns["inside_bar"] = min(inside_score, 1.0)

    if len(df) >= 3:
        a = df.iloc[-3]
        bullish = (
            a["close"] < a["open"]
            and prev["close"] < prev["open"]
            and last["close"] > last["open"]
            and last["close"] > max(a["high"], prev["high"])
        )
        bearish = (
            a["close"] > a["open"]
            and prev["close"] > prev["open"]
            and last["close"] < last["open"]
            and last["close"] < min(a["low"], prev["low"])
        )
        if bullish or bearish:
            patterns["three_bar_reversal"] = 1.0

    if vol_mean > 0:
        vol_score = last["volume"] / vol_mean - 1.0
        if vol_score > 0:
            patterns["volume_spike"] = min(vol_score, 1.0)

    if last["close"] >= high_max and last["volume"] > vol_mean:
        brk_score = min(1.0, last["volume"] / (vol_mean * 1.5))
        patterns["breakout"] = brk_score

    if last["close"] <= low_min and last["volume"] > vol_mean:
        brkdn_score = min(1.0, last["volume"] / (vol_mean * 1.5))
        patterns["breakdown"] = brkdn_score

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
