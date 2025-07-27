from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import ta
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model
    MODEL = load_model("momentum_bot")
except Exception:  # pragma: no cover - fallback
    MODEL = None


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
    """Donchian breakout with volume confirmation."""
    if df is None or df.empty:
        return 0.0, "none"

    params = config.get("momentum", {}) if config else {}
    window = int(params.get("donchian_window", 20))
    vol_window = int(params.get("volume_window", 20))
    vol_mult = float(params.get("volume_mult", 1.5))

    lookback = max(window, vol_window)
    if len(df) < lookback:
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

    dc_high = recent["high"].rolling(window).max().shift(1)
    dc_low = recent["low"].rolling(window).min().shift(1)
    vol_ma = recent["volume"].rolling(vol_window).mean()

    dc_high = cache_series("momentum_dc_high", df, dc_high, lookback)
    dc_low = cache_series("momentum_dc_low", df, dc_low, lookback)
    vol_ma = cache_series("momentum_vol_ma", df, vol_ma, lookback)

    recent = recent.copy()
    recent["dc_high"] = dc_high
    recent["dc_low"] = dc_low
    recent["vol_ma"] = vol_ma

    close = recent["close"].iloc[-1]
    volume = recent["volume"].iloc[-1]

    long_cond = close > dc_high.iloc[-1]
    short_cond = close < dc_low.iloc[-1]
    vol_ok = vol_ma.iloc[-1] > 0 and volume > vol_ma.iloc[-1] * vol_mult

    score = 0.0
    direction = "none"
    if long_cond and vol_ok:
        score = 1.0
        direction = "long"
    elif short_cond and vol_ok:
        score = 1.0
        direction = "short"

    if score > 0:
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = MODEL.predict(df)
                score = (score + ml_score) / 2
            except Exception:
                pass
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(recent, score)

    return score, direction


class regime_filter:
    """Match trending and volatile regimes."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime in {"trending", "volatile"}
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
