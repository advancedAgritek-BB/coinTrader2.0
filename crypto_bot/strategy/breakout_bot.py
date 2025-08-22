from __future__ import annotations

"""Breakout strategy using Bollinger/Keltner squeeze and trend filters.

Configuration
-------------
The ``breakout`` section of the config may include:

``ema_window`` (int)
    Window length for the EMA trend filter (default ``200``).
``adx_window`` (int)
    Lookback window for the ADX calculation (default ``14``).
``adx_threshold`` (float)
    Minimum ADX value required to qualify a breakout (default ``15``).

Existing options such as ``bb_length`` or ``donchian_window`` remain
unchanged.
"""

from typing import Optional, Tuple

import logging

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

from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils import stats
from crypto_bot.utils.ml_utils import init_ml_or_warn

NAME = "breakout_bot"
logger = logging.getLogger(__name__)

ML_AVAILABLE = init_ml_or_warn()

if ML_AVAILABLE:
    from coinTrader_Trainer.ml_trainer import load_model
    MODEL = load_model("breakout_bot")
else:  # pragma: no cover - fallback
    MODEL = None


def _squeeze(
    df: pd.DataFrame,
    bb_len: int,
    bb_std: float,
    kc_len: int,
    kc_mult: float,
    threshold: float,
    lookback: int,
    squeeze_pct: float,
) -> Tuple[pd.Series, pd.Series]:
    """Return squeeze boolean series and ATR values."""
    hist = max(bb_len, kc_len)
    if len(df) < hist + 1:
        return pd.Series(dtype=bool), pd.Series(dtype=float)
    recent = df.iloc[-(hist + 1):]

    close = recent["close"]
    high = recent["high"]
    low = recent["low"]

    bb = ta.volatility.BollingerBands(close, window=bb_len, window_dev=bb_std)
    bb_width = bb.bollinger_hband() - bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()

    atr = ta.volatility.average_true_range(high, low, close, window=kc_len)
    kc_width = 2 * atr * kc_mult

    if len(bb_width) >= lookback:
        width_z = stats.zscore(bb_width, lookback)
        thresh = scipy_stats.norm.ppf(squeeze_pct / 100)
        squeeze = (width_z < thresh) & (bb_width < kc_width)
    else:
        squeeze = (bb_width / bb_mid < threshold) & (bb_width < kc_width)

    bb_width = cache_series("bb_width", df, bb_width, hist)
    bb_mid = cache_series("bb_mid", df, bb_mid, hist)
    atr = cache_series("atr_kc", df, atr, hist)
    kc_width = cache_series("kc_width", df, kc_width, hist)
    squeeze = cache_series("squeeze", df, squeeze.astype(float), hist) > 0

    return squeeze, atr


def generate_signal(
    df: pd.DataFrame,
    symbol: str | None = None,
    timeframe: str | None = None,
    **kwargs,
) -> Tuple[float, str] | Tuple[float, str, float]:
    """Breakout strategy using Bollinger/Keltner squeeze confirmation.

    The ``config`` may contain a ``breakout`` section supporting additional
    options:

    - ``ema_window`` – window for the EMA trend filter (default ``200``)
    - ``adx_window`` – lookback window for ADX (default ``14``)
    - ``adx_threshold`` – minimum ADX value required (default ``15``)

    Returns
    -------
    Tuple[float, str] or Tuple[float, str, float]
        If ``higher_df`` is provided the function returns ``(score, direction)``.
        Otherwise it returns ``(score, direction, atr)`` where ``atr`` is the
        most recent Average True Range value.
    """
    if isinstance(symbol, dict) and timeframe is None:
        kwargs.setdefault("config", symbol)
        symbol = None
    if isinstance(timeframe, dict):
        kwargs.setdefault("config", timeframe)
        timeframe = None
    config = kwargs.get("config")
    higher_df: Optional[pd.DataFrame] = kwargs.get("higher_df")

    if df is None or df.empty:
        return (0.0, "none") if higher_df is not None else (0.0, "none", 0.0)

    cfg_all = config or {}
    cfg = cfg_all.get("breakout", {})
    bb_len = int(cfg.get("bb_length", 12))
    bb_std = float(cfg.get("bb_std", 2))
    kc_len = int(cfg.get("kc_length", 12))
    kc_mult = float(cfg.get("kc_mult", 1.5))
    donchian_window = int(cfg.get("donchian_window", cfg.get("dc_length", 30)))
    atr_buffer_mult = float(cfg.get("atr_buffer_mult", 0.05))
    vol_window = int(cfg.get("volume_window", 20))
    ema_window = int(cfg.get("ema_window", 200))
    adx_window = int(cfg.get("adx_window", 14))
    adx_threshold = float(cfg.get("adx_threshold", 15))
    score_threshold = float(cfg.get("score_threshold", 0.0))
    # Volume confirmation is now optional by default to allow breakouts without
    # a prior volume spike. Users can enable the old behaviour by explicitly
    # setting ``vol_confirmation: true`` in the strategy config.
    vol_confirmation = bool(cfg.get("vol_confirmation", False))
    vol_multiplier = float(
        cfg.get("vol_multiplier", cfg.get("volume_mult", 1.2))
    )
    threshold = float(cfg.get("squeeze_threshold", 0.03))
    lookback_cfg = int(cfg_all.get("indicator_lookback", 250))
    squeeze_pct = float(cfg_all.get("bb_squeeze_pct", 20))

    min_bars = max(100, adx_window + 1)
    lookback = max(
        bb_len,
        kc_len,
        donchian_window,
        vol_window,
        ema_window,
        adx_window,
        14,
        min_bars,
    )
    if len(df) < lookback:
        return (0.0, "none") if higher_df is not None else (0.0, "none", 0.0)

    recent = df.iloc[-(lookback + 1):]
    if len(recent) < min_bars:
        return (0.0, "none", 0.0) if higher_df is None else (0.0, "none")

    squeeze, atr = _squeeze(
        recent,
        bb_len,
        bb_std,
        kc_len,
        kc_mult,
        threshold,
        lookback_cfg,
        squeeze_pct,
    )
    if squeeze.empty or atr.empty:
        return (0.0, "none") if higher_df is not None else (0.0, "none", 0.0)
    if pd.isna(squeeze.iloc[-1]) or not squeeze.iloc[-1]:
        return (0.0, "none") if higher_df is not None else (0.0, "none", 0.0)

    if higher_df is not None and not higher_df.empty:
        higher_recent = higher_df.iloc[-(lookback + 1):]
        if len(higher_recent) < lookback:
            return (0.0, "none")
        _squeeze(
            higher_recent,
            bb_len,
            bb_std,
            kc_len,
            kc_mult,
            threshold,
            lookback_cfg,
            squeeze_pct,
        )
        # Higher timeframe squeeze is informative but no longer mandatory

    close = recent["close"]
    high = recent["high"]
    low = recent["low"]
    volume = recent["volume"]

    dc_high = high.rolling(donchian_window).max().shift(1)
    dc_low = low.rolling(donchian_window).min().shift(1)
    vol_ma = volume.rolling(vol_window).mean()

    rsi = ta.momentum.rsi(close, window=14)
    macd_hist = ta.trend.macd_diff(close)
    ema = ta.trend.ema_indicator(close, window=ema_window)
    adx = ta.trend.ADXIndicator(high, low, close, window=adx_window).adx()

    dc_high = cache_series("dc_high", df, dc_high, lookback)
    dc_low = cache_series("dc_low", df, dc_low, lookback)
    vol_ma = cache_series("vol_ma_breakout", df, vol_ma, lookback)
    rsi = cache_series("rsi_breakout", df, rsi, lookback)
    macd_hist = cache_series("macd_hist", df, macd_hist, lookback)
    ema = cache_series("ema_breakout", df, ema, lookback)
    adx = cache_series("adx_breakout", df, adx, lookback)

    recent = recent.copy()
    recent["dc_high"] = dc_high
    recent["dc_low"] = dc_low
    recent["vol_ma"] = vol_ma
    recent["rsi"] = rsi
    recent["macd_hist"] = macd_hist
    recent["ema"] = ema
    recent["adx"] = adx

    latest = recent.iloc[-1]

    vol_ma_last = vol_ma.iloc[-1] if not vol_ma.empty else float("nan")
    vol_last = volume.iloc[-1] if not volume.empty else float("nan")

    if vol_confirmation:
        vol_ok = vol_ma_last > 0 and vol_last > vol_ma_last * vol_multiplier
    else:
        vol_ok = True

    if (
        pd.isna(vol_ma_last)
        or pd.isna(vol_last)
        or vol_ma_last == 0
        or vol_last == 0
    ):
        vol_ok = False
    atr_last = atr.iloc[-1]
    upper_break = dc_high.iloc[-1] + atr_last * atr_buffer_mult
    lower_break = dc_low.iloc[-1] - atr_last * atr_buffer_mult

    metric = 0.0
    if latest["close"] > upper_break:
        metric = (latest["close"] - upper_break) / atr_last if atr_last > 0 else latest["close"] - upper_break
    elif latest["close"] < lower_break:
        metric = (latest["close"] - lower_break) / atr_last if atr_last > 0 else latest["close"] - lower_break

    direction = "none"
    if (
        vol_ok
        and abs(metric) > score_threshold
        and latest["adx"] >= adx_threshold
        and (
            (metric > 0 and latest["close"] > latest["ema"]) or
            (metric < 0 and latest["close"] < latest["ema"])
        )
    ):
        direction = "long" if metric > 0 else "short"

    score = metric
    if direction != "none" and MODEL is not None:
        try:  # pragma: no cover - best effort
            ml_score = MODEL.predict(df)
            score = (score + ml_score) / 2
        except Exception:
            pass
    if direction != "none" and (config is None or config.get("atr_normalization", True)):
        score = normalize_score_by_volatility(recent, score)

    logger.info(
        f"{df.index[-1]} Squeeze: {squeeze.iloc[-1]}, metric: {metric:.4f}, vol_ok: {vol_ok}"
    )

    if higher_df is not None:
        return score, direction
    return score, direction, atr_last


class regime_filter:
    """Match breakout regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "breakout"
