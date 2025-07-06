import pandas as pd


def detect_patterns(df: pd.DataFrame) -> dict[str, float]:
    """Return confidence scores for simple chart patterns detected in ``df``.
    """Return scored chart patterns detected in ``df``.

    The function inspects the latest candles and assigns a confidence
    between ``0`` and ``1`` for each detected pattern. Patterns include
    classic candlestick formations and simple breakout signals:

    ``"breakout"``         -- last close at a new high with volume spike
    ``"breakdown"``        -- last close at a new low with volume spike
    ``"hammer"``           -- small body, long lower shadow
    ``"shooting_star"``    -- small body, long upper shadow
    ``"doji"``             -- open and close nearly equal
    ``"bullish_engulfing"``  -- last candle engulfs previous bearish body
    ``"bearish_engulfing"`` -- last candle engulfs previous bullish body
    ``"inside_bar"``       -- current range is inside previous bar
    ``"three_bar_reversal"`` -- two bars one way then strong reversal
    ``"volume_spike"``     -- volume significantly above average
    """

    """Return a mapping of detected chart patterns to their strength.

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
    patterns: dict[str, float] = {}
    if df is None or len(df) < 2:
        return scores

    last = df.iloc[-1]

    body = abs(last["close"] - last["open"])
    candle_range = last["high"] - last["low"]
    if candle_range == 0:
        return scores
        return patterns

    upper = last["high"] - max(last["close"], last["open"])
    lower = min(last["close"], last["open"]) - last["low"]

    body_ratio = body / candle_range
    upper_ratio = upper / candle_range
    lower_ratio = lower / candle_range

    # hammer and shooting star scoring
    if body_ratio <= 0.4:
        hammer_score = max(0.0, min((lower_ratio - upper_ratio) * (1 - body_ratio), 1.0))
        shooting_score = max(0.0, min((upper_ratio - lower_ratio) * (1 - body_ratio), 1.0))
        if hammer_score > 0:
            patterns["hammer"] = hammer_score
        if shooting_score > 0:
            patterns["shooting_star"] = shooting_score

    # doji score based on body proportion
    doji_score = max(0.0, 1.0 - body_ratio / 0.1)
    if doji_score > 0:
        patterns["doji"] = min(doji_score, 1.0)
    if body <= candle_range * 0.1:
        if upper > candle_range * 0.4 and lower <= candle_range * 0.1:
            scores["shooting_star"] = 1.0
        if lower > candle_range * 0.4 and upper <= candle_range * 0.1:
            scores["hammer"] = 1.0
        if upper > candle_range * 0.2 and lower > candle_range * 0.2:
            scores["doji"] = 1.0
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

    # engulfing patterns
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

    # inside bar
    if last["high"] < prev["high"] and last["low"] > prev["low"]:
        range_prev = prev["high"] - prev["low"]
        inside_score = 1.0 - (prev["high"] - last["high"] + last["low"] - prev["low"]) / (range_prev + 1e-9)
        patterns["inside_bar"] = max(0.0, min(inside_score, 1.0))

    # three bar reversal
    if len(df) >= 3:
        a = df.iloc[-3]
        if (
            a["close"] < a["open"]
            and prev["close"] < prev["open"]
            and last["close"] > last["open"]
            and last["close"] > max(a["high"], prev["high"])
        ) or (
            a["close"] > a["open"]
            and prev["close"] > prev["open"]
            and last["close"] < last["open"]
            and last["close"] < min(a["low"], prev["low"])
        ):
            patterns["three_bar_reversal"] = 1.0

    # volume spike
    if vol_mean > 0:
        vol_score = max(0.0, min(last["volume"] / vol_mean - 1.0, 1.0))
        if vol_score > 0:
            patterns["volume_spike"] = vol_score

    # breakout and breakdown
    if last["close"] >= high_max and last["volume"] > vol_mean:
        brk_score = min(1.0, last["volume"] / (vol_mean * 1.5))
        patterns["breakout"] = max(brk_score, patterns.get("breakout", 0.0))

    if last["close"] <= low_min and last["volume"] > vol_mean:
        brkdn_score = min(1.0, last["volume"] / (vol_mean * 1.5))
        patterns["breakdown"] = max(brkdn_score, patterns.get("breakdown", 0.0))
    if last["close"] >= high_max and last["volume"] > vol_mean * 1.5:
        scores["breakout"] = 1.0
    if last["close"] <= low_min and last["volume"] > vol_mean * 1.5:
        scores["breakdown"] = 1.0
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

    return scores
