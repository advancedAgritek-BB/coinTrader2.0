from typing import Optional, Tuple

import pandas as pd
import ta
from crypto_bot.utils.indicator_cache import cache_series

from crypto_bot.utils.volatility import normalize_score_by_volatility


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

    lookback = max(rsi_window, vol_window, down_candles + 1)
    if len(df) < lookback:
        return 0.0, "none"

    recent = df.iloc[-(lookback + 1) :]

    rsi = ta.momentum.rsi(recent["close"], window=rsi_window)
    vol_ma = recent["volume"].rolling(window=vol_window).mean()

    rsi = cache_series("rsi", df, rsi, lookback)
    vol_ma = cache_series("vol_ma", df, vol_ma, lookback)

    df = recent.copy()
    df["rsi"] = rsi
    df["vol_ma"] = vol_ma

    latest = df.iloc[-1]
    prev_close = df["close"].iloc[-2]
    volume_spike = (
        latest["volume"] > latest["vol_ma"] * volume_multiple if latest["vol_ma"] > 0 else False
    )

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
    ):
        rsi_score = min((latest["rsi"] - overbought) / (100 - overbought), 1.0)
        vol_score = min(latest["volume"] / (latest["vol_ma"] * volume_multiple), 1.0)
        score = (rsi_score + vol_score) / 2
        direction = "short"

    if score > 0 and (config is None or config.get("atr_normalization", True)):
        score = normalize_score_by_volatility(df, score)

    return score, direction
