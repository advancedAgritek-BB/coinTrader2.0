"""Simple grid trading signal."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from crypto_bot.utils.volatility import normalize_score_by_volatility
from . import breakout_bot


@dataclass
class GridConfig:
    """Configuration options for :func:`generate_signal`."""

    num_levels: int = 5
    breakout_mult: float = 1.5
    atr_normalization: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "GridConfig":
        cfg = cfg or {}
        params = {}
        for f in fields(cls):
            if f.name == "num_levels":
                params[f.name] = int(cfg.get("num_levels", _get_num_levels()))
            else:
                params[f.name] = cfg.get(f.name, getattr(cls, f.name))
        return cls(**params)


ConfigType = Union[dict, GridConfig, None]


def _as_dict(cfg: ConfigType) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, GridConfig):
        return asdict(cfg)
    return dict(cfg)


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

    cfg_dict = _as_dict(config)
    cfg = GridConfig.from_dict(cfg_dict)

    if num_levels is None:
        num_levels = cfg.num_levels

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

    # defer to breakout logic when price extends far beyond the grid range
    breakout_threshold = half_range * cfg.breakout_mult
    if price > centre + breakout_threshold or price < centre - breakout_threshold:
        return breakout_bot.generate_signal(df, _as_dict(config))

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
