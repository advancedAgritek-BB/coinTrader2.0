from typing import Optional, Tuple

import pandas as pd
import ta

from crypto_bot.utils.volatility import normalize_score_by_volatility


def is_engulfing(df: pd.DataFrame, body_pct: float) -> Optional[str]:
    """Return ``"bullish"`` or ``"bearish"`` if the last candle engulfs the
    previous one and its body is at least ``body_pct`` of the range."""
    if len(df) < 2:
        return None

    prev = df.iloc[-2]
    last = df.iloc[-1]

    rng = last["high"] - last["low"]
    if rng <= 0:
        return None

    body_ratio = abs(last["close"] - last["open"]) / rng
    if body_ratio < body_pct:
        return None

    if (
        last["close"] > last["open"]
        and prev["close"] < prev["open"]
        and last["open"] <= prev["close"]
        and last["close"] >= prev["open"]
    ):
        return "bullish"

    if (
        last["close"] < last["open"]
        and prev["close"] > prev["open"]
        and last["open"] >= prev["close"]
        and last["close"] <= prev["open"]
    ):
        return "bearish"

    return None


def is_hammer(df: pd.DataFrame, body_pct: float) -> Optional[str]:
    """Return ``"bullish"`` or ``"bearish"`` if the last candle is a hammer
    with body at least ``body_pct`` of its range."""
    if df.empty:
        return None

    last = df.iloc[-1]
    rng = last["high"] - last["low"]
    if rng <= 0:
        return None

    body = abs(last["close"] - last["open"])
    body_ratio = body / rng
    if body_ratio < body_pct:
        return None

    upper = last["high"] - max(last["close"], last["open"])
    lower = min(last["close"], last["open"]) - last["low"]

    if lower > upper * 2:
        return "bullish"
    if upper > lower * 2:
        return "bearish"

    return None


def confirm_higher_lows(df: pd.DataFrame, bars: int) -> bool:
    """Return ``True`` if the last ``bars`` lows are strictly increasing."""
    if bars <= 1 or len(df) < bars:
        return True

    lows = df["low"].iloc[-bars:]
    return lows.diff().dropna().gt(0).all()


def confirm_lower_highs(df: pd.DataFrame, bars: int) -> bool:
    """Return ``True`` if the last ``bars`` highs are strictly decreasing."""
    if bars <= 1 or len(df) < bars:
        return True

    highs = df["high"].iloc[-bars:]
    return highs.diff().dropna().lt(0).all()


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Identify short-term bounces with volume confirmation."""
    if df.empty:
        return 0.0, "none"

    cfg = config or {}
    rsi_window = int(cfg.get("rsi_window", 14))
    oversold = float(cfg.get("oversold", 30))
    overbought = float(cfg.get("overbought", 70))
    vol_window = int(cfg.get("vol_window", 20))
    volume_multiple = float(cfg.get("volume_multiple", 2.0))
    down_candles = int(cfg.get("down_candles", 3))
    up_candles = int(cfg.get("up_candles", 3))
    body_pct = float(cfg.get("body_pct", 0.5))

    lookback = max(rsi_window, vol_window, down_candles + 1, up_candles + 1, 2)
    if len(df) < lookback:
        return 0.0, "none"

    df = df.copy()
    df["rsi"] = ta.momentum.rsi(df["close"], window=rsi_window)
    df["vol_ma"] = df["volume"].rolling(window=vol_window).mean()

    latest = df.iloc[-1]
    prev_close = df["close"].iloc[-2]
    volume_spike = (
        latest["volume"] > latest["vol_ma"] * volume_multiple if latest["vol_ma"] > 0 else False
    )

    eng_type = is_engulfing(df, body_pct)
    hammer_type = is_hammer(df, body_pct)
    bull_pattern = eng_type == "bullish" or hammer_type == "bullish"
    bear_pattern = eng_type == "bearish" or hammer_type == "bearish"

    higher_lows = confirm_higher_lows(df, up_candles)
    lower_highs = confirm_lower_highs(df, up_candles)

    recent_changes = df["close"].diff()
    downs = (recent_changes.iloc[-down_candles - 1 : -1] < 0).all()
    ups = (recent_changes.iloc[-down_candles - 1 : -1] > 0).all()

    score = 0.0
    direction = "none"

    if (
        not pd.isna(latest["rsi"])
        and downs
        and latest["close"] > prev_close
        and latest["rsi"] < oversold
        and volume_spike
        and bull_pattern
        and higher_lows
    ):
        rsi_score = min((oversold - latest["rsi"]) / oversold, 1.0)
        vol_score = min(latest["volume"] / (latest["vol_ma"] * volume_multiple), 1.0)
        score = (rsi_score + vol_score) / 2
        direction = "long"
    elif (
        not pd.isna(latest["rsi"])
        and ups
        and latest["close"] < prev_close
        and latest["rsi"] > overbought
        and volume_spike
        and bear_pattern
        and lower_highs
    ):
        rsi_score = min((latest["rsi"] - overbought) / (100 - overbought), 1.0)
        vol_score = min(latest["volume"] / (latest["vol_ma"] * volume_multiple), 1.0)
        score = (rsi_score + vol_score) / 2
        direction = "short"

    if score > 0 and (config is None or config.get("atr_normalization", True)):
        score = normalize_score_by_volatility(df, score)

    return score, direction
