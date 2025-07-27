from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import ta


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Simple momentum strategy using EMA divergence."""
    if df is None or df.empty:
        return 0.0, "none"

    cfg = config or {}
    fast = int(cfg.get("momentum_ema_fast", 5))
    slow = int(cfg.get("momentum_ema_slow", 20))

    ema_fast = ta.trend.ema_indicator(df["close"], window=fast)
    ema_slow = ta.trend.ema_indicator(df["close"], window=slow)

    latest_fast = ema_fast.iloc[-1]
    latest_slow = ema_slow.iloc[-1]
    latest_price = df["close"].iloc[-1]

    score = 0.0
    if latest_price:
        score = min(abs(latest_fast - latest_slow) / latest_price, 1.0)

    if latest_fast > latest_slow:
        return score, "long"
    if latest_fast < latest_slow:
        return score, "short"
    return 0.0, "none"
from crypto_bot.utils.indicator_cache import cache_series


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str]:
    """Simple Donchian breakout strategy with volume confirmation."""
    if df is None or df.empty:
        return 0.0, "none"

    params = config or {}
    dc_window = int(params.get("donchian_window", 20))
    vol_window = int(params.get("volume_window", 20))
    vol_mult = float(params.get("volume_mult", 1.5))

    lookback = max(dc_window, vol_window)
    if len(df) < lookback + 1:
        return 0.0, "none"

    recent = df.iloc[-(lookback + 1) :]

    high = recent["high"]
    low = recent["low"]
    close = recent["close"]
    volume = recent["volume"]

    dc_high = high.rolling(dc_window).max().shift(1)
    vol_ma = volume.rolling(vol_window).mean()

    dc_high = cache_series("dc_high_mom", df, dc_high, lookback)
    vol_ma = cache_series("vol_ma_mom", df, vol_ma, lookback)

    recent = recent.copy()
    recent["dc_high"] = dc_high
    recent["vol_ma"] = vol_ma

    last = recent.iloc[-1]
    breakout = last["close"] > last["dc_high"]
    vol_spike = (
        last["volume"] > last["vol_ma"] * vol_mult if last["vol_ma"] > 0 else False
    )

    if breakout and vol_spike:
        score = min((last["close"] - last["dc_high"]) / last["close"], 1.0)
        score = max(score, 0.0)
        return score, "long"

    return 0.0, "none"


class regime_filter:
    """Match momentum trading regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "momentum"
