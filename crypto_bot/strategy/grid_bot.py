"""Simple grid trading signal."""

from __future__ import annotations

import os
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import ta

from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility


def _get_num_levels() -> int:
    """Return grid levels from ``GRID_LEVELS`` env var or default of 5."""
    env = os.getenv("GRID_LEVELS")
    try:
        return int(env) if env else 5
    except ValueError:  # pragma: no cover - invalid env
        return 5


def recent_window(df: pd.DataFrame, cfg: Mapping[str, int] | None) -> pd.DataFrame:
    """Return trailing slice for indicator calculations."""
    params = cfg or {}
    range_window = int(params.get("range_window", 20))
    atr_period = int(params.get("atr_period", 14))
    volume_ma_window = int(params.get("volume_ma_window", 20))
    trend_ema_slow = int(params.get("trend_ema_slow", 50))
    lookback = max(range_window, atr_period, volume_ma_window, trend_ema_slow)
    return df.iloc[-lookback:]


def compute_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    """Return rolling Volume Weighted Average Price."""
    if not {"high", "low", "close", "volume"}.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    vwap = pv.rolling(window).sum() / df["volume"].rolling(window).sum()
    return vwap


def generate_signal(df: pd.DataFrame, num_levels: int | None = None, config: Optional[dict] = None) -> Tuple[float, str]:
    """Generate a grid based trading signal.

    The last 20 bars define a high/low range which is divided into grid levels.
    A positive score is returned when price trades near the lower grid levels
    and a negative score near the upper levels. The magnitude is proportional to
    the distance from the mid-point of the range. ``(0.0, "none")`` is returned
    when price stays around the centre of the grid.
    """

    if num_levels is None:
        num_levels = _get_num_levels()

    cfg = config or {}

    if df.empty:
        return 0.0, "none"

    recent = recent_window(df, cfg)
    if recent.empty:
        return 0.0, "none"

    range_window = int(cfg.get("range_window", 20))
    atr_period = int(cfg.get("atr_period", 14))
    volume_ma_window = int(cfg.get("volume_ma_window", 20))

    atr_series = ta.volatility.average_true_range(
        recent["high"], recent["low"], recent["close"], window=atr_period
    )
    vwap_series = compute_vwap(recent, volume_ma_window)

    lookback = len(recent)
    atr_series = cache_series(f"atr_{atr_period}", df, atr_series, lookback)
    vwap_series = cache_series(f"vwap_{volume_ma_window}", df, vwap_series, lookback)

    recent = recent.copy()
    recent["atr"] = atr_series
    recent["vwap"] = vwap_series

    range_slice = recent.iloc[-range_window:]
    high = range_slice["high"].max()
    low = range_slice["low"].min()

    if high == low:
        return 0.0, "none"

    price = recent["close"].iloc[-1]
    levels = np.linspace(low, high, num=num_levels)
    centre = (high + low) / 2
    half_range = (high - low) / 2

    lower_bound = levels[1]
    upper_bound = levels[-2]

    if price <= lower_bound:
        distance = centre - price
        score = min(distance / half_range, 1.0)
        if config is None or config.get('atr_normalization', True):
            score = normalize_score_by_volatility(df, score)
        return score, "long"
    if price >= upper_bound:
        distance = price - centre
        score = min(distance / half_range, 1.0)
        if config is None or config.get('atr_normalization', True):
            score = normalize_score_by_volatility(df, score)
        return score, "short"

    return 0.0, "none"
