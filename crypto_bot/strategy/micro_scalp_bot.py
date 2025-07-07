from typing import Optional, Tuple

import pandas as pd
import ta
from crypto_bot.utils.indicator_cache import cache_series

from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Return short-term signal using EMA crossover on 1m data.

    Parameters
    ----------
    df : pd.DataFrame
        Minute level OHLCV data.
    config : dict, optional
        Optional configuration overriding defaults located under
        ``micro_scalp`` in ``config.yaml``.
    """
    if df.empty:
        return 0.0, "none"

    params = config.get("micro_scalp", {}) if config else {}
    fast_window = int(params.get("ema_fast", 3))
    slow_window = int(params.get("ema_slow", 8))
    vol_window = int(params.get("volume_window", 20))
    vol_threshold = float(params.get("volume_threshold", 0))

    if len(df) < slow_window:
        return 0.0, "none"

    lookback = slow_window
    recent = df.iloc[-(lookback + 1) :]

    ema_fast = ta.trend.ema_indicator(recent["close"], window=fast_window)
    ema_slow = ta.trend.ema_indicator(recent["close"], window=slow_window)

    ema_fast = cache_series("ema_fast", df, ema_fast, lookback)
    ema_slow = cache_series("ema_slow", df, ema_slow, lookback)

    df = recent.copy()
    df["ema_fast"] = ema_fast
    df["ema_slow"] = ema_slow

    latest = df.iloc[-1]
    if pd.isna(latest["ema_fast"]) or pd.isna(latest["ema_slow"]):
        return 0.0, "none"

    if vol_threshold and "volume" in df.columns:
        vol_ma_series = df["volume"].rolling(vol_window).mean()
        vol_ma_series = cache_series("volume_ma", df, vol_ma_series, lookback)
        vol_ma = vol_ma_series.iloc[-1]
        if pd.isna(vol_ma) or vol_ma == 0 or latest["volume"] < vol_ma * vol_threshold:
            return 0.0, "none"

    momentum = latest["ema_fast"] - latest["ema_slow"]
    if momentum == 0:
        return 0.0, "none"

    score = min(abs(momentum) / latest["close"], 1.0)
    if config is None or config.get("atr_normalization", True):
        score = normalize_score_by_volatility(df, score)

    direction = "long" if momentum > 0 else "short"
    return score, direction
