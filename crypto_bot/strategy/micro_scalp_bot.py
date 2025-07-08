from typing import Optional, Tuple

import pandas as pd
import ta
from crypto_bot.utils.indicator_cache import cache_series

from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    higher_df: pd.DataFrame | None = None,
) -> Tuple[float, str]:
    """Return short-term signal using EMA crossover on 1m data.

    Parameters
    ----------
    df : pd.DataFrame
        Minute level OHLCV data.
    config : dict, optional
        Optional configuration overriding defaults located under
        ``micro_scalp`` in ``config.yaml``.
    higher_df : pd.DataFrame, optional
        Higher timeframe OHLCV data used to confirm the trend. When provided
        the function only returns a signal if ``trend_fast`` is above
        ``trend_slow`` for longs (and vice versa for shorts).
    """
    if df.empty:
        return 0.0, "none"

    params = config.get("micro_scalp", {}) if config else {}
    fast_window = int(params.get("ema_fast", 3))
    slow_window = int(params.get("ema_slow", 8))
    vol_window = int(params.get("volume_window", 20))
    vol_threshold = float(params.get("volume_threshold", 0))
    min_momentum_pct = float(params.get("min_momentum_pct", 0))
    confirm_bars = int(params.get("confirm_bars", 1))
    fresh_cross_only = bool(params.get("fresh_cross_only", False))
    trend_fast = int(params.get("trend_fast", 0))
    trend_slow = int(params.get("trend_slow", 0))
    _ = params.get("trend_timeframe")

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
    df["momentum"] = df["ema_fast"] - df["ema_slow"]

    latest = df.iloc[-1]
    if pd.isna(latest["ema_fast"]) or pd.isna(latest["ema_slow"]):
        return 0.0, "none"

    if vol_threshold and "volume" in df.columns:
        vol_ma_series = df["volume"].rolling(vol_window).mean()
        vol_ma_series = cache_series("volume_ma", df, vol_ma_series, lookback)
        vol_ma = vol_ma_series.iloc[-1]
        if pd.isna(vol_ma) or vol_ma == 0 or latest["volume"] < vol_ma * vol_threshold:
            return 0.0, "none"

    trend_fast_val = None
    trend_slow_val = None
    if higher_df is not None and trend_fast and trend_slow:
        trend_lookback = max(trend_fast, trend_slow)
        h_recent = higher_df.iloc[-(trend_lookback + 1) :]
        t_fast = ta.trend.ema_indicator(h_recent["close"], window=trend_fast)
        t_slow = ta.trend.ema_indicator(h_recent["close"], window=trend_slow)
        t_fast = cache_series("trend_fast", higher_df, t_fast, trend_lookback)
        t_slow = cache_series("trend_slow", higher_df, t_slow, trend_lookback)
        trend_fast_val = t_fast.iloc[-1]
        trend_slow_val = t_slow.iloc[-1]
        if pd.isna(trend_fast_val) or pd.isna(trend_slow_val):
            return 0.0, "none"

    momentum = df["momentum"].iloc[-1]
    if momentum == 0:
        return 0.0, "none"

    if min_momentum_pct and abs(momentum) / latest["close"] < min_momentum_pct:
        return 0.0, "none"

    if confirm_bars > 0:
        if len(df) < confirm_bars:
            return 0.0, "none"
        signs = (df["momentum"].iloc[-confirm_bars:].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0))
        if signs.abs().min() == 0 or not (signs == signs.iloc[-1]).all():
            return 0.0, "none"
        if fresh_cross_only and len(df) >= confirm_bars + 1:
            prev_sign = 1 if df["momentum"].iloc[-confirm_bars - 1] > 0 else -1 if df["momentum"].iloc[-confirm_bars - 1] < 0 else 0
            if prev_sign == signs.iloc[-1]:
                return 0.0, "none"
        elif fresh_cross_only and len(df) < confirm_bars + 1:
            return 0.0, "none"

    score = min(abs(momentum) / latest["close"], 1.0)
    if config is None or config.get("atr_normalization", True):
        score = normalize_score_by_volatility(df, score)

    direction = "long" if momentum > 0 else "short"

    if trend_fast_val is not None and trend_slow_val is not None:
        if direction == "long" and trend_fast_val <= trend_slow_val:
            return 0.0, "none"
        if direction == "short" and trend_fast_val >= trend_slow_val:
            return 0.0, "none"

    return score, direction
