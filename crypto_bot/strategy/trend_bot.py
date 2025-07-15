from typing import Optional, Tuple

import pandas as pd
import numpy as np
import ta
try:  # pragma: no cover - optional dependency
    from scipy import stats as scipy_stats
    if not hasattr(scipy_stats, "norm"):
        raise ImportError
except Exception:  # pragma: no cover - fallback when scipy missing
    class _Norm:
        @staticmethod
        def ppf(_x):
            return 0.0

    class _FakeStats:
        norm = _Norm()

    scipy_stats = _FakeStats()
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils import stats

from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Trend following signal with ADX, volume and optional Donchian filters."""
    if df.empty or len(df) < 50:
        return 0.0, "none"

    df = df.copy()
    params = config or {}
    lookback_cfg = int(params.get("indicator_lookback", 250))
    rsi_overbought_pct = float(params.get("rsi_overbought_pct", 90))
    rsi_oversold_pct = float(params.get("rsi_oversold_pct", 10))
    fast_window = int(params.get("trend_ema_fast", 3))
    slow_window = int(params.get("trend_ema_slow", 10))
    atr_period = int(params.get("atr_period", 14))
    k = float(params.get("k", 1.0))
    volume_window = int(params.get("volume_window", 20))
    volume_mult = float(params.get("volume_mult", 1.0))
    adx_threshold = float(params.get("adx_threshold", 25))

    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=fast_window)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=slow_window)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["rsi_z"] = stats.zscore(df["rsi"], lookback_cfg)
    df["volume_ma"] = df["volume"].rolling(window=volume_window).mean()
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_period)
    lookback = max(50, volume_window)
    recent = df.iloc[-(lookback + 1) :]

    ema20 = ta.trend.ema_indicator(recent["close"], window=20)
    ema50 = ta.trend.ema_indicator(recent["close"], window=50)
    rsi = ta.momentum.rsi(recent["close"], window=14)
    vol_ma = recent["volume"].rolling(window=volume_window).mean()

    ema20 = cache_series("ema20", df, ema20, lookback)
    ema50 = cache_series("ema50", df, ema50, lookback)
    rsi = cache_series("rsi", df, rsi, lookback)
    vol_ma = cache_series(f"volume_ma_{volume_window}", df, vol_ma, lookback)

    df = recent.copy()
    df["ema20"] = ema20
    df["ema50"] = ema50
    df["rsi"] = rsi
    df["rsi_z"] = stats.zscore(rsi, lookback_cfg)
    df["volume_ma"] = vol_ma

    adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=7)
    df["adx"] = adx_ind.adx()

    latest = df.iloc[-1]
    score = 0.0
    direction = "none"

    atr_pct = 0.0
    if latest["close"] != 0:
        atr_pct = (latest["atr"] / latest["close"]) * 100
    dynamic_oversold = min(90.0, 30 + k * atr_pct)
    dynamic_overbought = max(10.0, 70 - k * atr_pct)

    rsi_z_last = df["rsi_z"].iloc[-1]
    rsi_z_series = df["rsi_z"].dropna()
    if not rsi_z_series.empty:
        upper_thr = rsi_z_series.quantile(rsi_overbought_pct / 100)
        lower_thr = rsi_z_series.quantile(rsi_oversold_pct / 100)
    else:
        upper_thr = np.nan
        lower_thr = np.nan

    overbought_cond = (
        rsi_z_last > upper_thr
        if not pd.isna(upper_thr) and not pd.isna(rsi_z_last)
        else latest["rsi"] > dynamic_overbought
    )
    oversold_cond = (
        rsi_z_last < lower_thr
        if not pd.isna(lower_thr) and not pd.isna(rsi_z_last)
        else latest["rsi"] < dynamic_oversold
    )

    upper_thr = scipy_stats.norm.ppf(rsi_overbought_pct / 100)
    lower_thr = scipy_stats.norm.ppf(rsi_oversold_pct / 100)
    volume_ok = latest["volume"] > latest["volume_ma"] * volume_mult
    overbought_cond = (
        (
            rsi_z_last > upper_thr
            if not pd.isna(rsi_z_last)
            else latest["rsi"] > dynamic_overbought
        )
        and volume_ok
    )
    oversold_cond = (
        (
            rsi_z_last < lower_thr
            if not pd.isna(rsi_z_last)
            else latest["rsi"] < dynamic_oversold
        )
        and volume_ok
    )

    long_cond = (
        latest["close"] > latest["ema_fast"]
        and latest["ema_fast"] > latest["ema_slow"]
        and overbought_cond
        and latest["adx"] > adx_threshold
        and latest["volume"] > latest["volume_ma"]
    )
    short_cond = (
        latest["close"] < latest["ema_fast"]
        and latest["ema_fast"] < latest["ema_slow"]
        and oversold_cond
        and latest["adx"] > adx_threshold
        and latest["volume"] > latest["volume_ma"]
    )

    if params.get("donchian_confirmation", False):
        window = params.get("donchian_window", 20)
        upper = df["high"].rolling(window=window).max().iloc[-1]
        lower = df["low"].rolling(window=window).min().iloc[-1]
        long_cond = long_cond and latest["close"] >= upper
        short_cond = short_cond and latest["close"] <= lower

    cross_up = (
        df["ema_fast"].iloc[-2] < df["ema_slow"].iloc[-2]
        and latest["ema_fast"] > latest["ema_slow"]
    )
    cross_down = (
        df["ema_fast"].iloc[-2] > df["ema_slow"].iloc[-2]
        and latest["ema_fast"] < latest["ema_slow"]
    )
    reversal_long = cross_up and oversold_cond and latest["volume"] > latest["volume_ma"]
    reversal_short = cross_down and overbought_cond and latest["volume"] > latest["volume_ma"]

    if long_cond:
        score = min((latest["rsi"] - 50) / 50, 1.0)
        direction = "long"
    elif short_cond:
        score = min((50 - latest["rsi"]) / 50, 1.0)
        direction = "short"
    elif reversal_long:
        score = min((dynamic_oversold - latest["rsi"]) / dynamic_oversold, 1.0)
        direction = "long"
    elif reversal_short:
        score = min((latest["rsi"] - dynamic_overbought) / (100 - dynamic_overbought), 1.0)
        direction = "short"

    if score > 0 and (config is None or config.get("atr_normalization", True)):
        score = normalize_score_by_volatility(df, score)

    if config:
        torch_cfg = config.get("torch_signal_model", {})
        if torch_cfg.get("enabled") and score > 0:
            weight = float(torch_cfg.get("weight", 0.7))
            try:  # pragma: no cover - best effort
                from crypto_bot.torch_signal_model import predict_signal as _pred
                ml_score = _pred(df)
                score = score * (1 - weight) + ml_score * weight
                score = max(0.0, min(score, 1.0))
            except Exception:
                pass

    return score, direction


class regime_filter:
    """Match trending regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "trending"
