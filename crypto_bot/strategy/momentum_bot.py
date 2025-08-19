from typing import Optional, Tuple

import logging
import pandas as pd
import ta
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once

logger = logging.getLogger(__name__)

NAME = "momentum_bot"

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model
    ML_AVAILABLE = True
except Exception:  # pragma: no cover - trainer missing
    ML_AVAILABLE = False

MODEL: Optional[object]
if ML_AVAILABLE:
    MODEL = load_model("momentum_bot")
else:  # pragma: no cover - fallback
    MODEL = None
    warn_ml_unavailable_once()


def generate_signal(
    df: pd.DataFrame,
    config: dict | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    **_,
) -> Tuple[float, str]:
    """Donchian breakout with volume confirmation."""
    if df is None or df.empty:
        return 0.0, "none"

    params = config.get("momentum_bot", {}) if config else {}
    window = int(params.get("donchian_window", 20))
    vol_window = int(params.get("volume_window", 20))
    vol_mult = float(params.get("volume_mult", 1.5))
    rsi_threshold = float(params.get("rsi_threshold", 55))
    macd_min = float(params.get("macd_min", 0.0))
    macd_fast = int(params.get("fast_length", 12))
    macd_slow = int(params.get("slow_length", 26))
    rsi_window = 14

    min_len = max(window, vol_window, macd_slow, rsi_window)
    if len(df) < min_len:
        return 0.0, "none"

    lookback = min(len(df), min_len)
    recent = df.iloc[-(lookback + 1) :]

    dc_low = recent["low"].rolling(window).min().shift(1)
    vol_ma = recent["volume"].rolling(vol_window).mean()
    rsi = ta.momentum.rsi(recent["close"], window=rsi_window)
    macd = ta.trend.macd(
        recent["close"], window_fast=macd_fast, window_slow=macd_slow
    )

    dc_low = cache_series("momentum_dc_low", df, dc_low, lookback)
    vol_ma = cache_series("momentum_vol_ma", df, vol_ma, lookback)
    rsi = cache_series("momentum_rsi", df, rsi, lookback)
    macd = cache_series("momentum_macd", df, macd, lookback)

    recent = recent.copy()
    recent["dc_low"] = dc_low
    recent["vol_ma"] = vol_ma
    recent["rsi"] = rsi
    recent["macd"] = macd

    latest = recent.iloc[-1]

    macd_val = latest["macd"]
    rsi_val = latest["rsi"]

    score = 0.0
    direction = "none"

    long_cond = macd_val > 0 or rsi_val > 50
    short_cond = (
        latest["close"] < dc_low.iloc[-1]
        and latest["rsi"] < 100 - rsi_threshold
        and latest["macd"] < -macd_min
    )
    vol_ok = (
        pd.notna(latest["vol_ma"])
        and latest["vol_ma"] > 0
        and latest["volume"] > latest["vol_ma"] * vol_mult
    )

    if long_cond:
        score = 0.8
        direction = "long"
        logger.info(
            f"momentum_bot long signal: MACD={macd_val}, RSI={rsi_val}"
        )
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

    logger.info(
        "RSI %.2f MACD %.5f score %.2f direction %s",
        float(latest.get("rsi", float("nan"))),
        float(latest.get("macd", float("nan"))),
        score,
        direction,
    )
    return score, direction


class regime_filter:
    """Match trending and volatile regimes."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime in {"trending", "volatile"}
