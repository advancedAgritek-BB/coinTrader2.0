from typing import Optional, Tuple

import numpy as np

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
from ta.trend import ADXIndicator
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils import stats

from crypto_bot.utils.volatility import normalize_score_by_volatility


def generate_signal(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[float, str]:
    """Score mean reversion opportunities using multiple indicators."""

    if len(df) < 50:
        return 0.0, "none"

    params = config or {}
    lookback_cfg = int(params.get("indicator_lookback", 14))
    rsi_overbought_pct = float(params.get("rsi_overbought_pct", 90))
    rsi_oversold_pct = float(params.get("rsi_oversold_pct", 10))
    adx_threshold = float(params.get("adx_threshold", 20))
    sl_mult = float(params.get("sl_mult", 1.5))
    tp_mult = float(params.get("tp_mult", 2.0))
    ml_enabled = bool(params.get("ml_enabled", True))

    lookback = 14
    recent = df.iloc[-(lookback + 1) :]

    rsi = ta.momentum.rsi(recent["close"], window=14)
    rsi_z = stats.zscore(rsi, lookback_cfg)
    mean = recent["close"].rolling(14).mean()
    std = recent["close"].rolling(14).std()
    bb_z = (recent["close"] - mean) / std

    kc = ta.volatility.KeltnerChannel(
        recent["high"], recent["low"], recent["close"], window=14
    )
    kc_h = kc.keltner_channel_hband()
    kc_l = kc.keltner_channel_lband()

    bb_full = ta.volatility.BollingerBands(df["close"], window=14)
    bb_width_full = bb_full.bollinger_wband()
    median_bw_20_full = bb_width_full.rolling(14).median()
    bb_width = bb_width_full.iloc[-(lookback + 1) :]
    median_bw_20 = median_bw_20_full.iloc[-(lookback + 1) :]

    vwap = ta.volume.VolumeWeightedAveragePrice(
        recent["high"], recent["low"], recent["close"], recent["volume"], window=14
    )
    vwap_series = vwap.volume_weighted_average_price()
    atr_full = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    adx_full = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    atr = atr_full.iloc[-(lookback + 1) :]
    adx = adx_full.iloc[-(lookback + 1) :]

    adx = ta.trend.ADXIndicator(
        df["high"], df["low"], df["close"], window=14
    ).adx()

    rsi = cache_series("rsi", df, rsi, lookback)
    rsi_z = cache_series("rsi_z", df, rsi_z, lookback)
    bb_z = cache_series("bb_z", df, bb_z, lookback)
    bb_width = cache_series("bb_width", df, bb_width, lookback)
    median_bw_20 = cache_series("median_bw_20", df, median_bw_20, lookback)
    kc_h = cache_series("kc_h", df, kc_h, lookback)
    kc_l = cache_series("kc_l", df, kc_l, lookback)
    vwap_series = cache_series("vwap", df, vwap_series, lookback)
    atr = cache_series("atr", df, atr, lookback)
    adx = cache_series("adx", df, adx, lookback)

    df = recent.copy()
    df["rsi"] = rsi
    df["rsi_z"] = rsi_z
    df["bb_z"] = bb_z
    df["bb_width"] = bb_width
    df["median_bw_20"] = median_bw_20
    df["kc_h"] = kc_h
    df["kc_l"] = kc_l
    df["vwap"] = vwap_series
    df["atr"] = atr
    df["adx"] = adx

    width_series = (df["kc_h"] - df["kc_l"]).dropna()
    if len(width_series) >= lookback:
        median_width = width_series.iloc[-lookback:].median()
        if width_series.iloc[-1] > median_width:
            return 0.0, "none"

    latest = df.iloc[-1]

    if (
        pd.isna(df["bb_width"].iloc[-1])
        or pd.isna(df["median_bw_20"].iloc[-1])
        or df["bb_width"].iloc[-1] >= df["median_bw_20"].iloc[-1]
    ):
        return 0.0, "none"

    if df["adx"].iloc[-1] > adx_threshold:
        return 0.0, "none"

    long_scores = []
    short_scores = []

    rsi_z_last = df["rsi_z"].iloc[-1]
    atr_pct = 0.0 if latest["close"] == 0 else (latest["atr"] / latest["close"]) * 100
    dynamic_oversold_pct = np.clip(rsi_oversold_pct + atr_pct * sl_mult, 1, 49)
    dynamic_overbought_pct = np.clip(rsi_overbought_pct - atr_pct * tp_mult, 51, 99)
    lower_thr = scipy_stats.norm.ppf(dynamic_oversold_pct / 100)
    upper_thr = scipy_stats.norm.ppf(dynamic_overbought_pct / 100)
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

    if ml_enabled:
        try:
            from crypto_bot.ml_signal_model import predict_signal
            ml_score = predict_signal(df)
            score = (score + ml_score) / 2
        except Exception:
            pass

    if config is None or config.get("atr_normalization", True):
        score = normalize_score_by_volatility(df, score)

    return float(max(0.0, min(score, 1.0))), direction


class regime_filter:
    """Match mean-reverting regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "mean-reverting"
