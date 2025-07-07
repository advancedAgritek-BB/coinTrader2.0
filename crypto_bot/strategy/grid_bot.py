"""Simple grid trading signal."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, asdict
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from dataclasses import asdict, dataclass, is_dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from crypto_bot.volatility_filter import calc_atr
import ta
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot import grid_state


@dataclass
class GridConfig:
    """Configuration options for :func:`generate_signal`."""

    cooldown_bars: int = 6
    max_active_legs: int = 4
    symbol: str = ""
    atr_normalization: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "GridConfig":
        cfg = cfg or {}
        params = {f.name: cfg.get(f.name, getattr(cls, f.name)) for f in fields(cls)}
        return cls(**params)


def _as_dict(cfg: Union[dict, GridConfig, None]) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, GridConfig):
        return asdict(cfg)
    return dict(cfg)


@dataclass
class GridConfig:
    """Configuration values for :func:`generate_signal`."""

    atr_normalization: bool = True
    volume_ma_window: int = 20
    volume_multiple: float = 1.5
    vol_zscore_threshold: float = 2.0
    """Configuration for :func:`generate_signal`."""

    atr_period: int = 14
    spacing_factor: float = 0.75
    """Configuration options for :func:`generate_signal`."""

    trend_ema_fast: int = 50
    trend_ema_slow: int = 200
    atr_normalization: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "GridConfig":
        cfg = cfg or {}
        params = {f.name: cfg.get(f.name, getattr(cls, f.name)) for f in fields(cls)}
        return cls(**params)


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
def _as_dict(cfg: ConfigType) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, GridConfig):
        return asdict(cfg)
    return dict(cfg)
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


def generate_signal(
    df: pd.DataFrame,
    num_levels: int | None = None,
    config: Optional[Union[dict, GridConfig]] = None,
    config: ConfigType = None,
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

    cfg_dict = _as_dict(config)
    cfg = GridConfig.from_dict(cfg_dict)

    symbol = cfg.symbol
    if symbol:
        grid_state.update_bar(symbol, len(df))
        if grid_state.in_cooldown(symbol, cfg.cooldown_bars):
            return 0.0, "none"
        if grid_state.active_leg_count(symbol) >= cfg.max_active_legs:
            return 0.0, "none"
    cfg = _get_config(config)

    min_len = max(20, cfg.volume_ma_window)
    if df.empty or len(df) < min_len or "volume" not in df:
        return 0.0, "none"

    if not volume_ok(df["volume"], cfg.volume_ma_window, cfg.volume_multiple, cfg.vol_zscore_threshold):
    cfg_dict = _as_dict(config)
    cfg = GridConfig.from_dict(cfg_dict)
    atr_period = cfg.atr_period
    spacing_factor = cfg.spacing_factor

    lookback = max(20, atr_period)
    if df.empty or len(df) < lookback:
        return 0.0, "none"

    recent = df.tail(lookback)
    cfg = GridConfig.from_dict(config)

    if df.empty or len(df) < 20:
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
    centre = recent["vwap"].iloc[-1] if "vwap" in recent.columns else (high + low) / 2

    atr = calc_atr(recent, window=atr_period)
    grid_step = atr * spacing_factor
    if not grid_step:
        return 0.0, "none"

    n = num_levels // 2
    levels = centre + np.arange(-n, n + 1) * grid_step
    half_range = grid_step * n

    lower_bound = levels[1]
    upper_bound = levels[-2]

    if price <= lower_bound:
        if not is_in_trend(recent, cfg.trend_ema_fast, cfg.trend_ema_slow, "long"):
            return 0.0, "none"
        distance = centre - price
        score = min(distance / half_range, 1.0)
        if cfg.atr_normalization:
        if cfg.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        return score, "long"

    if price >= upper_bound:
        if not is_in_trend(recent, cfg.trend_ema_fast, cfg.trend_ema_slow, "short"):
            return 0.0, "none"
        distance = price - centre
        score = min(distance / half_range, 1.0)
        if cfg.atr_normalization:
        if cfg.get("atr_normalization", True):
            score = normalize_score_by_volatility(df, score)
        return score, "short"

    return 0.0, "none"
