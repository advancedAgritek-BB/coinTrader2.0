from typing import Optional, Tuple

import logging
import pandas as pd
import ta
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.ml_utils import init_ml_or_warn

logger = logging.getLogger(__name__)

NAME = "momentum_bot"

ML_AVAILABLE = init_ml_or_warn()
MODEL: Optional[object]
if ML_AVAILABLE:
    from coinTrader_Trainer.ml_trainer import load_model
    MODEL = load_model("momentum_bot")
else:  # pragma: no cover - fallback
    MODEL = None


def generate_signal(
    df: pd.DataFrame,
    symbol: str | None = None,
    timeframe: str | None = None,
    **kwargs,
) -> Tuple[float, str]:
    """Donchian breakout with volume confirmation."""
    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config")

    if df is None or df.empty:
        return 0.0, "none"

    params = config.get("momentum_bot", {}) if config else {}
    window = int(params.get("donchian_window", 20))
    vol_window = int(params.get("volume_window", 20))
    vol_z_min = float(params.get("volume_z_min", 0.5))
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
    vol_std = recent["volume"].rolling(vol_window).std()
    rsi = ta.momentum.rsi(recent["close"], window=rsi_window)
    macd = ta.trend.macd(
        recent["close"], window_fast=macd_fast, window_slow=macd_slow
    )

    dc_low = cache_series("momentum_dc_low", df, dc_low, lookback)
    vol_ma = cache_series("momentum_vol_ma", df, vol_ma, lookback)
    vol_std = cache_series("momentum_vol_std", df, vol_std, lookback)
    volume_z = cache_series(
        "momentum_volume_z", df, (recent["volume"] - vol_ma) / vol_std, lookback
    )
    rsi = cache_series("momentum_rsi", df, rsi, lookback)
    macd = cache_series("momentum_macd", df, macd, lookback)

    recent = recent.copy()
    recent["dc_low"] = dc_low
    recent["vol_ma"] = vol_ma
    recent["vol_std"] = vol_std
    recent["volume_z"] = volume_z
    recent["rsi"] = rsi
    recent["macd"] = macd

    latest = recent.iloc[-1]

    macd_val = latest["macd"]
    rsi_val = latest["rsi"]
    volume_z = latest["volume_z"]

    score = 0.0
    direction = "none"

    long_cond = macd_val > 0 or rsi_val > 50
    short_cond = (
        latest["close"] < dc_low.iloc[-1]
        and latest["rsi"] < 100 - rsi_threshold
        and latest["macd"] < -macd_min
    )

    if long_cond:
        score = 0.8
        direction = "long"
        logger.info(
            f"momentum_bot long signal: MACD={macd_val}, RSI={rsi_val}"
        )
    elif short_cond and volume_z > vol_z_min:
        score = min(1.0, volume_z / 2)
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
