from typing import Optional, Tuple

import pandas as pd
import ta
try:  # pragma: no cover - optional dependency
    from scipy import stats as scipy_stats
    if not hasattr(scipy_stats, "norm"):
        raise ImportError
except Exception:  # pragma: no cover - fallback
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
    """Score mean reversion opportunities using multiple indicators."""

    if len(df) < 3:
        return 0.0, "none"

    params = config or {}
    lookback_cfg = int(params.get("indicator_lookback", 250))
    rsi_overbought_pct = float(params.get("rsi_overbought_pct", 90))
    rsi_oversold_pct = float(params.get("rsi_oversold_pct", 10))

    lookback = 20
    recent = df.iloc[-(lookback + 1) :]

    rsi = ta.momentum.rsi(recent["close"], window=14)
    rsi_z = stats.zscore(rsi, lookback_cfg)
    mean = recent["close"].rolling(20).mean()
    std = recent["close"].rolling(20).std()
    bb_z = (recent["close"] - mean) / std

    kc = ta.volatility.KeltnerChannel(
        recent["high"], recent["low"], recent["close"], window=20
    )
    kc_h = kc.keltner_channel_hband()
    kc_l = kc.keltner_channel_lband()

    vwap = ta.volume.VolumeWeightedAveragePrice(
        recent["high"], recent["low"], recent["close"], recent["volume"], window=14
    )
    vwap_series = vwap.volume_weighted_average_price()

    rsi = cache_series("rsi", df, rsi, lookback)
    rsi_z = cache_series("rsi_z", df, rsi_z, lookback)
    bb_z = cache_series("bb_z", df, bb_z, lookback)
    kc_h = cache_series("kc_h", df, kc_h, lookback)
    kc_l = cache_series("kc_l", df, kc_l, lookback)
    vwap_series = cache_series("vwap", df, vwap_series, lookback)

    df = recent.copy()
    df["rsi"] = rsi
    df["rsi_z"] = rsi_z
    df["bb_z"] = bb_z
    df["kc_h"] = kc_h
    df["kc_l"] = kc_l
    df["vwap"] = vwap_series

    latest = df.iloc[-1]

    long_scores = []
    short_scores = []

    rsi_z_last = df["rsi_z"].iloc[-1]
    lower_thr = scipy_stats.norm.ppf(rsi_oversold_pct / 100)
    upper_thr = scipy_stats.norm.ppf(rsi_overbought_pct / 100)
    oversold_cond = (
        rsi_z_last < lower_thr if not pd.isna(rsi_z_last) else latest["rsi"] < 50
    )
    overbought_cond = (
        rsi_z_last > upper_thr if not pd.isna(rsi_z_last) else latest["rsi"] > 50
    )

    if oversold_cond:
        long_scores.append(min((50 - latest["rsi"]) / 20, 1))
    elif overbought_cond:
        short_scores.append(min((latest["rsi"] - 50) / 20, 1))

    if not pd.isna(latest["bb_z"]):
        if latest["bb_z"] < 0:
            long_scores.append(min(-latest["bb_z"] / 2, 1))
        elif latest["bb_z"] > 0:
            short_scores.append(min(latest["bb_z"] / 2, 1))

    if not pd.isna(latest["kc_h"]) and not pd.isna(latest["kc_l"]):
        width = latest["kc_h"] - latest["kc_l"]
        if width > 0:
            if latest["close"] < latest["kc_l"]:
                long_scores.append(min((latest["kc_l"] - latest["close"]) / width, 1))
            elif latest["close"] > latest["kc_h"]:
                short_scores.append(min((latest["close"] - latest["kc_h"]) / width, 1))

    if not pd.isna(latest["vwap"]):
        diff = abs(latest["close"] - latest["vwap"]) / latest["vwap"]
        if latest["close"] < latest["vwap"]:
            long_scores.append(min(diff / 0.05, 1))
        elif latest["close"] > latest["vwap"]:
            short_scores.append(min(diff / 0.05, 1))

    long_score = sum(long_scores) / len(long_scores) if long_scores else 0.0
    short_score = sum(short_scores) / len(short_scores) if short_scores else 0.0

    if long_score > short_score and long_score > 0:
        score = long_score
        direction = "long"
    elif short_score > long_score and short_score > 0:
        score = short_score
        direction = "short"
    else:
        return 0.0, "none"

    if config is None or config.get("atr_normalization", True):
        score = normalize_score_by_volatility(df, score)

    return float(max(0.0, min(score, 1.0))), direction
