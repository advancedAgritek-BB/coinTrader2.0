"""Simple grid trading signal."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import ta
from crypto_bot.utils.volatility import normalize_score_by_volatility


@dataclass
class GridConfig:
    """Configuration options for :func:`generate_signal`."""

    trend_ema_fast: int = 50
    trend_ema_slow: int = 200
    atr_normalization: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "GridConfig":
        cfg = cfg or {}
        return cls(
            trend_ema_fast=int(cfg.get("trend_ema_fast", cls.trend_ema_fast)),
            trend_ema_slow=int(cfg.get("trend_ema_slow", cls.trend_ema_slow)),
            atr_normalization=bool(
                cfg.get("atr_normalization", cls.atr_normalization)
            ),
        )


def _get_num_levels() -> int:
    """Return grid levels from ``GRID_LEVELS`` env var or default of 5."""
    env = os.getenv("GRID_LEVELS")
    try:
        return int(env) if env else 5
    except ValueError:  # pragma: no cover - invalid env
        return 5


def is_in_trend(df: pd.DataFrame, fast: int, slow: int, side: str) -> bool:
    """Return ``True`` if ``side`` aligns with the EMA trend."""

    if len(df) < max(fast, slow):
        return True

    fast_ema = ta.trend.ema_indicator(df["close"], window=fast).iloc[-1]
    slow_ema = ta.trend.ema_indicator(df["close"], window=slow).iloc[-1]

    if pd.isna(fast_ema) or pd.isna(slow_ema):
        return True

    if side == "long":
        return fast_ema > slow_ema
    if side == "short":
        return fast_ema < slow_ema
    return True


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

    cfg = GridConfig.from_dict(config)

    if df.empty or len(df) < 20:
        return 0.0, "none"

    recent = df.tail(20)
    high = recent["high"].max()
    low = recent["low"].min()

    if high == low:
        return 0.0, "none"

    price = recent["close"].iloc[-1]
    levels = np.linspace(low, high, num=num_levels)
    centre = (high + low) / 2
    half_range = (high - low) / 2

    lower_bound = levels[1]
    upper_bound = levels[-2]

    if price <= lower_bound:
        if not is_in_trend(recent, cfg.trend_ema_fast, cfg.trend_ema_slow, "long"):
            return 0.0, "none"
        distance = centre - price
        score = min(distance / half_range, 1.0)
        if cfg.atr_normalization:
            score = normalize_score_by_volatility(df, score)
        return score, "long"

    if price >= upper_bound:
        if not is_in_trend(recent, cfg.trend_ema_fast, cfg.trend_ema_slow, "short"):
            return 0.0, "none"
        distance = price - centre
        score = min(distance / half_range, 1.0)
        if cfg.atr_normalization:
            score = normalize_score_by_volatility(df, score)
        return score, "short"

    return 0.0, "none"
