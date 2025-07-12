"""Simple grid trading signal."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ta

from crypto_bot import grid_state
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.volatility_filter import calc_atr
from . import breakout_bot


@dataclass
class GridConfig:
    """Configuration options for :func:`generate_signal`."""

    num_levels: int = 5
    breakout_mult: float = 1.5
    cooldown_bars: int = 6
    max_active_legs: int = 4
    symbol: str = ""

    atr_normalization: bool = True
    volume_ma_window: int = 20
    volume_multiple: float = 1.5
    vol_zscore_threshold: float = 2.0

    atr_period: int = 14
    spacing_factor: float = 0.75
    trend_ema_fast: int = 50
    trend_ema_slow: int = 200

    range_window: int = 20
    min_range_pct: float = 0.001

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
        data = asdict(cfg)
        env_levels = os.getenv("GRID_LEVELS")
        if env_levels and cfg.num_levels == GridConfig.num_levels:
            data.pop("num_levels", None)
        return data
    return dict(cfg)


def _get_num_levels() -> int:
    """Return grid levels from ``GRID_LEVELS`` env var or default of 5."""
    env = os.getenv("GRID_LEVELS")
    try:
        return int(env) if env else 5
    except ValueError:  # pragma: no cover - invalid env
        return 5


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


def recent_window(df: pd.DataFrame, cfg: Mapping[str, int] | None) -> pd.DataFrame:
    """Return trailing slice for indicator calculations."""
    params = cfg or {}
    range_window = int(params.get("range_window", 20))
    atr_period = int(params.get("atr_period", 14))
    volume_ma_window = int(params.get("volume_ma_window", 20))
    lookback = min(len(df), range_window)
    lookback = max(lookback, atr_period)
    return df.iloc[-lookback:]


def compute_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    """Return rolling Volume Weighted Average Price."""
    if not {"high", "low", "close", "volume"}.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    vol_sum = df["volume"].rolling(window).sum()
    price_sum = pv.rolling(window).sum()
    rolling_mean = typical.rolling(window, min_periods=1).mean()
    vwap = price_sum / vol_sum
    vwap = vwap.where(vol_sum != 0, rolling_mean)
    return vwap.fillna(rolling_mean)


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


def generate_signal(
    df: pd.DataFrame,
    num_levels: int | None = None,
    config: ConfigType = None,
) -> Tuple[float, str]:
    """Generate a grid based trading signal."""
    cfg = GridConfig.from_dict(_as_dict(config))

    if num_levels is None:
        num_levels = cfg.num_levels

    symbol = cfg.symbol
    if symbol:
        grid_state.update_bar(symbol, len(df))
        if grid_state.in_cooldown(symbol, cfg.cooldown_bars):
            return 0.0, "none"
        if grid_state.active_leg_count(symbol) >= cfg.max_active_legs:
            return 0.0, "none"

    min_len = max(20, cfg.volume_ma_window)
    if df.empty or len(df) < min_len:
        return 0.0, "none"

    if "volume" in df and not volume_ok(
        df["volume"], cfg.volume_ma_window, cfg.volume_multiple, cfg.vol_zscore_threshold
    ):
        return 0.0, "none"

    range_window = cfg.range_window
    atr_period = cfg.atr_period
    volume_ma_window = cfg.volume_ma_window

    recent_len = max(range_window, atr_period, volume_ma_window)
    recent = df.tail(recent_len)
    if recent.empty:
        return 0.0, "none"

    range_slice = df.tail(range_window)
    high = range_slice["high"].max()
    low = range_slice["low"].min()

    if high == low:
        return 0.0, "none"

    price = recent["close"].iloc[-1]
    range_size = high - low
    if range_size < price * cfg.min_range_pct:
        return 0.0, "none"

    atr_series = ta.volatility.average_true_range(
        recent["high"], recent["low"], recent["close"], window=atr_period
    )
    vwap_series = compute_vwap(recent, volume_ma_window)

    lookback = len(recent)
    atr_series = cache_series(f"atr_{atr_period}", df, atr_series, lookback)
    vwap_series = cache_series(f"vwap_{volume_ma_window}", df, vwap_series, lookback)

    recent = recent.copy()
    recent["atr"] = atr_series
    if not vwap_series.isna().all():
        recent["vwap"] = vwap_series

    if "vwap" in recent.columns and not pd.isna(recent["vwap"].iloc[-1]):
        centre = recent["vwap"].iloc[-1]
    else:
        centre = (high + low) / 2

    atr = calc_atr(recent, window=atr_period)
    range_step = (high - low) / max(num_levels - 1, 1)
    grid_step = min(atr * cfg.spacing_factor, range_step)
    if not grid_step:
        return 0.0, "none"

    n = num_levels // 2
    levels = centre + np.arange(-n, n + 1) * grid_step
    half_range = grid_step * n
    breakout_range = (high - low) / 2
    breakout_threshold = breakout_range * cfg.breakout_mult
    if price > centre + breakout_threshold or price < centre - breakout_threshold:
        return breakout_bot.generate_signal(df, _as_dict(config))

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
