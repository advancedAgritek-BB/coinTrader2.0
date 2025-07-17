import pandas as pd
import numpy as np

try:  # pragma: no cover - optional scipy dependency
    from scipy.signal import find_peaks  # type: ignore
except Exception:  # pragma: no cover - fallback when scipy missing

    def find_peaks(data, distance=1, *a, **k):
        arr = np.asarray(data)
        idx = []
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                idx.append(i)
        return np.array(idx), {}


def detect_patterns(df: pd.DataFrame, *, min_conf: float = 0.0) -> dict[str, float]:
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

    ema = df["close"].ewm(span=20, adjust=False).mean()
    ema_slope = float(ema.diff().iloc[-1]) if len(ema) > 1 else 0.0

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
        if ema_slope < 0:
            hammer_score *= 1.2
        if hammer_score > 0:
            patterns["hammer"] = min(hammer_score, 1.0)

        shooting_score = max(0.0, (upper_ratio - lower_ratio) * (1 - body_ratio))
        if ema_slope > 0:
            shooting_score *= 1.2
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

    if vol_mean > 0 and last["close"] >= high_max and last["volume"] > vol_mean:
        brk_score = min(1.0, last["volume"] / (vol_mean * 1.5))
        patterns["breakout"] = brk_score

    if vol_mean > 0 and last["close"] <= low_min and last["volume"] > vol_mean:
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

    # Head and shoulders pattern using peak detection
    hs_lookback = min(len(df), 25)
    series = df["close"].iloc[-hs_lookback:]
    peaks, _ = find_peaks(series, distance=2)
    if len(peaks) >= 3:
        p1, p2, p3 = peaks[-3:]
        h1, h2, h3 = series.iloc[[p1, p2, p3]]
        if h2 > h1 and h2 > h3:
            tol = series.iloc[-1] * 0.02
            if abs(h1 - h3) <= tol and h2 - max(h1, h3) >= tol:
                if hs_lookback - 1 - p3 <= 3:
                    patterns["head_and_shoulders"] = 1.0

    troughs, _ = find_peaks(-series, distance=2)
    if len(troughs) >= 3:
        t1, t2, t3 = troughs[-3:]
        l1, l2, l3 = series.iloc[[t1, t2, t3]]
        if l2 < l1 and l2 < l3:
            tol = series.iloc[-1] * 0.02
            if abs(l1 - l3) <= tol and min(l1, l3) - l2 >= tol:
                if hs_lookback - 1 - t3 <= 3:
                    patterns["inverse_head_and_shoulders"] = 1.0

    return {k: v for k, v in patterns.items() if v >= min_conf}
