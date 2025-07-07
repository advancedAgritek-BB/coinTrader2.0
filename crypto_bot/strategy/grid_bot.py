"""Simple grid trading signal."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, is_dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from crypto_bot.utils.volatility import normalize_score_by_volatility


def _get_num_levels() -> int:
    """Return grid levels from ``GRID_LEVELS`` env var or default of 5."""
    env = os.getenv("GRID_LEVELS")
    try:
        return int(env) if env else 5
    except ValueError:  # pragma: no cover - invalid env
        return 5


@dataclass
class GridConfig:
    """Configuration for :func:`generate_signal`."""

    range_window: int = 20


def generate_signal(
    df: pd.DataFrame,
    num_levels: int | None = None,
    config: Optional[dict | GridConfig] = None,
) -> Tuple[float, str]:
    """Generate a grid based trading signal.

    The recent ``range_window`` bars define a high/low range divided into grid levels.
    A positive score is returned when price trades near the lower grid levels
    and a negative score near the upper levels. The magnitude is proportional to
    the distance from the mid-point of the range. ``(0.0, "none")`` is returned
    when price stays around the centre of the grid.
    """

    if num_levels is None:
        num_levels = _get_num_levels()

    cfg = {}
    if config is not None:
        if is_dataclass(config):
            cfg = asdict(config)
        else:
            cfg = dict(config)

    window = int(cfg.get("range_window", 20))

    if df.empty or len(df) < window:
        return 0.0, "none"

    recent = df.tail(window)
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
        if cfg.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        return score, "long"
    if price >= upper_bound:
        distance = price - centre
        score = min(distance / half_range, 1.0)
        if cfg.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        return score, "short"

    return 0.0, "none"
