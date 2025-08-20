"""Trend (momentum) trading strategy.

This momentum bot expects ``exit_strategy.trailing_stop_factor`` to set
an ATR-based trailing stop for risk management.
"""

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
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once

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
    warn_ml_unavailable_once()

if ML_AVAILABLE:
    MODEL = load_model("trend_bot")
else:  # pragma: no cover - fallback
    MODEL = None


def generate_signal(
    df: pd.DataFrame,
    symbol: str | None = None,
    timeframe: str | None = None,
    **kwargs,
) -> Tuple[float, str]:
    """Trend following signal with ADX, volume and optional Donchian filters."""
    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config") or {}
    symbol = config.get("symbol", "")
    adx_window = 7
    min_bars = max(50, adx_window + 1)
    if df.empty or len(df) < min_bars:
        score_logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
        return 0.0, "none"

    df = df.copy()
    params = config or {}

    def _num(name, default, cast=float, min_value=None, max_value=None):
        """Safely read and clamp numeric parameters from the config."""
        raw = params.get(name, default)
        try:
            value = cast(raw)
        except (TypeError, ValueError):
            value = default
        if min_value is not None:
            value = max(min_value, value)
        if max_value is not None:
            value = min(max_value, value)
        return value

    lookback_cfg = _num("indicator_lookback", 250, int, 1)
    rsi_overbought_pct = _num("rsi_overbought_pct", 90.0, float, 0.0, 100.0)
    rsi_oversold_pct = _num("rsi_oversold_pct", 10.0, float, 0.0, 100.0)
    fast_window = _num("trend_ema_fast", 3, int, 1)
    slow_window = _num("trend_ema_slow", 10, int, 1)
    atr_period = _num("atr_period", 14, int, 1)
    k = _num("k", 1.0, float, 0.0)
    volume_window = _num("volume_window", 20, int, 1)
    volume_mult = _num("volume_mult", 1.0, float, 0.0)
    adx_threshold = _num("adx_threshold", 25.0, float, 0.0)

    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=fast_window)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=slow_window)
    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=atr_period
    )

    lookback = max(50, volume_window)

    rsi_full = ta.momentum.rsi(df["close"], window=14)
    rsi_full = cache_series("rsi", df, rsi_full, lookback)
    vol_ma_full = df["volume"].rolling(window=volume_window).mean()
    vol_ma_full = cache_series(f"volume_ma_{volume_window}", df, vol_ma_full, lookback)
    ema20_full = ta.trend.ema_indicator(df["close"], window=20)
    ema20_full = cache_series("ema20", df, ema20_full, lookback)
    ema50_full = ta.trend.ema_indicator(df["close"], window=50)
    ema50_full = cache_series("ema50", df, ema50_full, lookback)

    recent = df.iloc[-(lookback + 1) :].copy()
    recent["rsi"] = rsi_full
    recent["rsi_z"] = stats.zscore(recent["rsi"], lookback_cfg)
    recent["volume_ma"] = vol_ma_full
    recent["ema20"] = ema20_full
    recent["ema50"] = ema50_full
    df = recent

    df["adx"] = ta.trend.ADXIndicator(
        df["high"], df["low"], df["close"], window=7
    ).adx()
    df["adx"] = cache_series("adx_trend", df, df["adx"], lookback)

    latest = df.iloc[-1]
    if (
        pd.isna(latest["ema_fast"]) or pd.isna(latest["ema_slow"]) or pd.isna(latest["rsi"])
    ):
        score_logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
        return 0.0, "none"

    adx = float(latest["adx"])
    close = df["close"]
    ema_slow = df["ema_slow"]
    if adx > 15 and close.iloc[-1] > ema_slow.iloc[-1]:
        score = 0.8
        direction = "long"
        logger.info(
            "trend_bot: ADX=%s, EMA_slow=%s, score=%s", adx, ema_slow.iloc[-1], score
        )
        return score, direction

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

    # Derive overbought/oversold thresholds from the distribution of recent
    # RSI z-scores.  These quantile based levels adapt to the data and proved
    # more robust in practice than theoretical values from `norm.ppf`.  The
    # old behaviour using `scipy_stats.norm.ppf` has been removed but left
    # here for reference:
    #
    #    upper_thr_norm = scipy_stats.norm.ppf(rsi_overbought_pct / 100)
    #    lower_thr_norm = scipy_stats.norm.ppf(rsi_oversold_pct / 100)

    volume_ok = latest["volume"] > latest["volume_ma"] * volume_mult
    overbought_cond = (
        (
            rsi_z_last > upper_thr
            if not pd.isna(upper_thr) and not pd.isna(rsi_z_last)
            else latest["rsi"] > dynamic_overbought
        )
        and volume_ok
    )
    oversold_cond = (
        (
            rsi_z_last < lower_thr
            if not pd.isna(lower_thr) and not pd.isna(rsi_z_last)
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

    logger.debug(
        "long_cond=%s short_cond=%s close=%s ema_fast=%s ema_slow=%s rsi=%s adx=%s volume=%s volume_ma=%s",
        long_cond,
        short_cond,
        float(latest["close"]),
        float(latest["ema_fast"]),
        float(latest["ema_slow"]),
        float(latest["rsi"]),
        float(latest["adx"]),
        float(latest["volume"]),
        float(latest["volume_ma"]),
    )

    if config.get("donchian_confirmation", False):
        try:
            window = int(config.get("donchian_window", 20))
        except (TypeError, ValueError):
            window = 20
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

    if score > 0:
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = MODEL.predict(df)
                score = (score + ml_score) / 2
            except Exception:
                pass
        if config is None or config.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)

    if config:
        torch_cfg = config.get("torch_signal_model", {})
        if torch_cfg.get("enabled") and score > 0:
            try:
                weight = float(torch_cfg.get("weight", 0.7))
            except (TypeError, ValueError):
                weight = 0.7
            try:  # pragma: no cover - best effort
                from crypto_bot.torch_signal_model import predict_signal as _pred
                ml_score = _pred(df)
                score = score * (1 - weight) + ml_score * weight
                score = max(0.0, min(score, 1.0))
            except Exception:
                pass
    score_logger.info("Signal for %s: %s, %s", symbol, score, direction)
    return score, direction


class regime_filter:
    """Match trending regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "trending"


class Strategy:
    """Strategy wrapper so :func:`load_strategies` can auto-register it."""

    def __init__(self) -> None:
        self.name = "trend_bot"
        self.generate_signal = generate_signal
        self.regime_filter = regime_filter


__all__ = ["generate_signal", "regime_filter", "Strategy"]
