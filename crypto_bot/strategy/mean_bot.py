from typing import Optional, Tuple

import asyncio
import importlib

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
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once

from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.logger import LOG_DIR, setup_logger
NAME = "mean_bot"

logger = setup_logger(__name__, LOG_DIR / "bot.log")
# Shared logger for symbol scoring
score_logger = setup_logger(
    "symbol_filter", LOG_DIR / "symbol_filter.log", to_console=False
)

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model
    ML_AVAILABLE = True
except Exception:  # pragma: no cover - trainer missing
    ML_AVAILABLE = False

if ML_AVAILABLE:
    MODEL = load_model("mean_bot")
else:  # pragma: no cover - fallback
    MODEL = None
    warn_ml_unavailable_once()


async def generate_signal(
    df: pd.DataFrame,
    symbol: str | None = None,
    timeframe: str | None = None,
    **kwargs,
) -> Tuple[float, str]:
    """Score mean reversion opportunities using multiple indicators."""

    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config") or {}
    symbol = config.get("symbol", "")
    timeframe = timeframe or config.get("timeframe")
    adx_window = 14
    min_bars = max(50, adx_window + 1)
    if len(df) < min_bars:
        score_logger.info(
            "Signal for %s:%s -> %.3f, %s",
            symbol or "unknown",
            timeframe or "N/A",
            0.0,
            "none",
        )
        return 0.0, "none"

    try:
        lookback_cfg = int(config.get("indicator_lookback", 14))
    except (TypeError, ValueError):
        lookback_cfg = 14
    try:
        rsi_overbought_pct = float(config.get("rsi_overbought_pct", 65))
    except (TypeError, ValueError):
        rsi_overbought_pct = 65.0
    try:
        rsi_oversold_pct = float(config.get("rsi_oversold_pct", 35))
    except (TypeError, ValueError):
        rsi_oversold_pct = 35.0
    try:
        adx_threshold = float(config.get("adx_threshold", 25))
    except (TypeError, ValueError):
        adx_threshold = 25.0
    try:
        sl_mult = float(config.get("sl_mult", 1.5))
    except (TypeError, ValueError):
        sl_mult = 1.5
    try:
        tp_mult = float(config.get("tp_mult", 2.0))
    except (TypeError, ValueError):
        tp_mult = 2.0
    ml_enabled = bool(config.get("ml_enabled", True))

    lookback = 14

    rsi_full = ta.momentum.rsi(df["close"], window=14)
    rsi_full = cache_series("rsi", df, rsi_full, lookback)
    rsi_z_full = stats.zscore(rsi_full, lookback_cfg)
    rsi_z_full = cache_series("rsi_z", df, rsi_z_full, lookback)

    recent = df.iloc[-(lookback + 1) :].copy()

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

    bb_z = cache_series("bb_z", df, bb_z, lookback)
    bb_width = cache_series("bb_width", df, bb_width, lookback)
    median_bw_20 = cache_series("median_bw_20", df, median_bw_20, lookback)
    kc_h = cache_series("kc_h", df, kc_h, lookback)
    kc_l = cache_series("kc_l", df, kc_l, lookback)
    vwap_series = cache_series("vwap", df, vwap_series, lookback)
    atr = cache_series("atr", df, atr, lookback)
    adx = cache_series("adx", df, adx, lookback)

    df = recent.copy()
    df["rsi"] = rsi_full
    df["rsi_z"] = rsi_z_full
    df["bb_z"] = bb_z
    df["bb_width"] = bb_width
    df["median_bw_20"] = median_bw_20
    df["kc_h"] = kc_h
    df["kc_l"] = kc_l
    df["vwap"] = vwap_series
    df["atr"] = atr
    df["adx"] = adx

    df = df.dropna()
    if df.empty:
        return 0.0, "none"

    width_series = (df["kc_h"] - df["kc_l"]).dropna()
    if width_series.empty:
        return 0.0, "none"
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
        if df.isna().any().any():
            return score, direction

        ml_score = None
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = await asyncio.to_thread(MODEL.predict, df)
            except Exception:
                ml_score = None
        else:  # pragma: no cover - fallback
            try:
                ml_mod = importlib.import_module("crypto_bot.ml_signal_model")
                ml_score = ml_mod.predict_signal(df)
            except Exception:
                return score, direction

        if ml_score is not None:
            score = (score + ml_score) / 2

    if config is None or config.get("atr_normalization", True):
        score = normalize_score_by_volatility(df, score)

    score = float(max(0.0, min(score, 1.0)))
    score_logger.info(
        "Signal for %s:%s -> %.3f, %s",
        symbol or "unknown",
        timeframe or "N/A",
        score,
        direction,
    )
    return score, direction


class regime_filter:
    """Match mean-reverting regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "mean-reverting"
