from typing import Optional, Tuple

import pandas as pd
import ta
from crypto_bot.utils.indicator_cache import cache_series

from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Trend following signal with ADX, volume and optional Donchian filters."""
    if df.empty or len(df) < 50:
        return 0.0, "none"

    lookback = 50
    recent = df.iloc[-(lookback + 1) :]

    ema20 = ta.trend.ema_indicator(recent["close"], window=20)
    ema50 = ta.trend.ema_indicator(recent["close"], window=50)
    rsi = ta.momentum.rsi(recent["close"], window=14)
    vol_ma = recent["volume"].rolling(window=20).mean()

    ema20 = cache_series("ema20", df, ema20, lookback)
    ema50 = cache_series("ema50", df, ema50, lookback)
    rsi = cache_series("rsi", df, rsi, lookback)
    vol_ma = cache_series("volume_ma", df, vol_ma, lookback)

    df = recent.copy()
    df["ema20"] = ema20
    df["ema50"] = ema50
    df["rsi"] = rsi
    df["volume_ma"] = vol_ma

    adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"] = adx_ind.adx()

    latest = df.iloc[-1]
    score = 0.0
    direction = "none"

    long_cond = (
        latest["ema20"] > latest["ema50"]
        and latest["rsi"] > 55
        and latest["adx"] > 20
        and latest["volume"] > latest["volume_ma"]
    )
    short_cond = (
        latest["ema20"] < latest["ema50"]
        and latest["rsi"] < 45
        and latest["adx"] > 20
        and latest["volume"] > latest["volume_ma"]
    )

    if config and config.get("donchian_confirmation"):
        window = config.get("donchian_window", 20)
        upper = df["high"].rolling(window=window).max().iloc[-1]
        lower = df["low"].rolling(window=window).min().iloc[-1]
        long_cond = long_cond and latest["close"] >= upper
        short_cond = short_cond and latest["close"] <= lower

    if long_cond:
        score = min((latest["rsi"] - 50) / 50, 1.0)
        direction = "long"
    elif short_cond:
        score = min((50 - latest["rsi"]) / 50, 1.0)
        direction = "short"

    if score > 0 and (config is None or config.get("atr_normalization", True)):
        score = normalize_score_by_volatility(df, score)

    return score, direction
