"""Simple grid trading signal."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from crypto_bot.utils.volatility import normalize_score_by_volatility


@dataclass
class GridConfig:
    """Configuration values for :func:`generate_signal`."""

    atr_normalization: bool = True
    volume_ma_window: int = 20
    volume_multiple: float = 1.5
    vol_zscore_threshold: float = 2.0


ConfigType = Union[dict, GridConfig, None]


def _get_config(cfg: ConfigType) -> GridConfig:
    if cfg is None:
        return GridConfig()
    if isinstance(cfg, GridConfig):
        return cfg
    cfg = dict(cfg)
    return GridConfig(
        atr_normalization=cfg.get("atr_normalization", True),
        volume_ma_window=cfg.get("volume_ma_window", 20),
        volume_multiple=cfg.get("volume_multiple", 1.5),
        vol_zscore_threshold=cfg.get("vol_zscore_threshold", 2.0),
    )


def volume_ok(series: pd.Series, window: int, mult: float, z_thresh: float) -> bool:
    """Return ``True`` if ``series`` shows a volume spike."""

    if len(series) < window:
        return False

    recent = series.tail(window)
    mean = recent.mean()
    std = recent.std()

    if np.isnan(mean):
        return False

    current = series.iloc[-1]
    z = (current - mean) / std if std > 0 else float("-inf")
    return current > mean * mult or z >= z_thresh


def _get_num_levels() -> int:
    """Return grid levels from ``GRID_LEVELS`` env var or default of 5."""
    env = os.getenv("GRID_LEVELS")
    try:
        return int(env) if env else 5
    except ValueError:  # pragma: no cover - invalid env
        return 5


def generate_signal(
    df: pd.DataFrame,
    num_levels: int | None = None,
    config: ConfigType = None,
) -> Tuple[float, str]:
    """Generate a grid based trading signal.

    The last 20 bars define a high/low range which is divided into grid levels.
    A positive score is returned when price trades near the lower grid levels
    and a negative score near the upper levels. The magnitude is proportional to
    the distance from the mid-point of the range. ``(0.0, "none")`` is returned
    when price stays around the centre of the grid.
    """

    if num_levels is None:
        num_levels = _get_num_levels()

    cfg = _get_config(config)

    min_len = max(20, cfg.volume_ma_window)
    if df.empty or len(df) < min_len or "volume" not in df:
        return 0.0, "none"

    if not volume_ok(df["volume"], cfg.volume_ma_window, cfg.volume_multiple, cfg.vol_zscore_threshold):
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
        distance = centre - price
        score = min(distance / half_range, 1.0)
        if cfg.atr_normalization:
            score = normalize_score_by_volatility(df, score)
        return score, "long"
    if price >= upper_bound:
        distance = price - centre
        score = min(distance / half_range, 1.0)
        if cfg.atr_normalization:
            score = normalize_score_by_volatility(df, score)
        return score, "short"

    return 0.0, "none"
