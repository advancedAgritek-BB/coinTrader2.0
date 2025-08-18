"""Simple grid trading signal."""

from __future__ import annotations

import os
import dataclasses
from dataclasses import dataclass, field, fields, asdict
from typing import Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ta

from crypto_bot import grid_state
from crypto_bot.utils.indicator_cache import cache_series
from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.volatility import atr_percent
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.volatility_filter import calc_atr
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once

DYNAMIC_THRESHOLD = 1.5
logger = setup_logger(__name__, LOG_DIR / "bot.log")

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model
    ML_AVAILABLE = True
except Exception:  # pragma: no cover - trainer missing
    ML_AVAILABLE = False

# ``micro_scalp_bot`` is imported lazily inside ``generate_signal`` to avoid
# issues if the module has optional dependencies or is otherwise unavailable
# during import. Only ``breakout_bot`` is required at module load time.
from . import breakout_bot
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
from crypto_bot.utils.regime_pnl_tracker import get_recent_win_rate

if ML_AVAILABLE:
    MODEL = load_model("grid_bot")
else:  # pragma: no cover - fallback
    MODEL = None
    warn_ml_unavailable_once()


@dataclass
class GridConfig:
    """Configuration options for :func:`generate_signal`.

    Attributes include standard grid parameters along with options for dynamic
    grids, machine-learning centered ranges and simple arbitrage support.
    """

    num_levels: int = 6
    breakout_mult: float = 1.5
    cooldown_bars: int = 2
    max_active_legs: int = 8
    symbol: str = ""
    leverage: int = 1

    atr_normalization: bool = True
    atr_period: int = 14
    range_window: int = 20
    spacing_factor: float = 0.3
    dynamic_grid: bool = True
    use_ml_center: bool = False
    min_range_pct: float = 0.0005
    arbitrage_pairs: list[tuple[str, str]] = field(default_factory=list)
    arbitrage_threshold: float = 0.005
    trend_ema_fast: int = 50
    trend_ema_slow: int = 200
    volume_ma_window: int = 20
    volume_multiple: float = 1.5
    vol_zscore_threshold: float = 2.0
    atr_change_threshold: float = 0.1
    volume_filter: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "GridConfig":
        cfg = cfg or {}
        params = {}
        for f in fields(cls):
            if f.name == "num_levels":
                params[f.name] = int(cfg.get("num_levels", _get_num_levels()))
            else:
                if f.default is not dataclasses.MISSING:
                    default = f.default
                elif f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
                    default = f.default_factory()
                else:
                    default = None
                params[f.name] = cfg.get(f.name, default)
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
    """Return grid levels from ``GRID_LEVELS`` env var or default of 10."""
    env = os.getenv("GRID_LEVELS")
    try:
        return int(env) if env else 10
    except ValueError:  # pragma: no cover - invalid env
        return 10


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
    higher_df: pd.DataFrame | None = None,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    mempool_monitor: SolanaMempoolMonitor | None = None,
    mempool_cfg: dict | None = None,
    **_,
) -> Tuple[float, str]:
    """Generate a grid based trading signal."""
    cfg = GridConfig.from_dict(_as_dict(config))

    if num_levels is None:
        num_levels = cfg.num_levels

    symbol = cfg.symbol
    if symbol:
        grid_state.update_bar(symbol, len(df))
        win_rate = get_recent_win_rate(4, strategy="grid_bot")
        skip_cd = win_rate > 0.7
        if not skip_cd and grid_state.in_cooldown(symbol, cfg.cooldown_bars):
            logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
            return 0.0, "none"
        if grid_state.active_leg_count(symbol) >= cfg.max_active_legs:
            logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
            return 0.0, "none"

    min_len = max(20, cfg.volume_ma_window)
    if df.empty or len(df) < min_len:
        return 0.0, "none"

    if (
        cfg.volume_filter
        and "volume" in df
        and not volume_ok(
            df["volume"],
            cfg.volume_ma_window,
            cfg.volume_multiple,
            cfg.vol_zscore_threshold,
        )
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

    range_slice = df.tail(range_window)
    high = range_slice["high"].max()
    low = range_slice["low"].min()

    if high == low:
        return 0.0, "none"

    price = recent["close"].iloc[-1]
    if price == 0:
        return 0.0, "none"

    range_pct = (high - low) / price
    if cfg.min_range_pct and range_pct < cfg.min_range_pct:
        return 0.0, "none"
    if "vwap" in recent.columns and not pd.isna(recent["vwap"].iloc[-1]):
        centre = recent["vwap"].iloc[-1]
    else:
        centre = (high + low) / 2

    atr_pct_1h = atr_percent(higher_df or df, atr_period)
    spacing_pct = max(0.3, 1.2 * atr_pct_1h)
    grid_step = price * spacing_pct / 100

    if cfg.dynamic_grid and symbol:
        prev_step = grid_state.get_grid_step(symbol)
        prev_pct = grid_state.get_last_atr(symbol)
        if (
            prev_step is None
            or prev_pct is None
            or prev_pct == 0
            or abs(atr_pct_1h - prev_pct) / prev_pct > 0.2
        ):
            grid_state.set_grid_step(symbol, grid_step)
        else:
            grid_step = prev_step
        grid_state.set_last_atr(symbol, atr_pct_1h)
    # Determine the centre price of the grid. Previous versions optionally used
    # a machine learning model via ``grid_center_model.predict_centre``. Since
    # no model is shipped with the repository, fall back to a simple heuristic
    # based on the latest VWAP when available, otherwise the midpoint of the
    # recent high/low range.
    if "vwap" in recent.columns and not pd.isna(recent["vwap"].iloc[-1]):
        centre = recent["vwap"].iloc[-1]
    else:
        centre = (high + low) / 2

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

    if cfg.dynamic_grid and lower_bound < price < upper_bound:
        near_lower = price - lower_bound <= grid_step
        near_upper = upper_bound - price <= grid_step
        if (near_lower or near_upper) and {"open", "high", "low", "close"}.issubset(df.columns):
            try:  # pragma: no cover - optional dependency
                from . import micro_scalp_bot
            except Exception:  # pragma: no cover - best effort
                micro_scalp_bot = None

            if micro_scalp_bot is not None:
                scalp_score, scalp_dir = micro_scalp_bot.generate_signal(
                    df,
                    _as_dict(config),
                    higher_df=higher_df,
                    mempool_monitor=mempool_monitor,
                    mempool_cfg=mempool_cfg,
                )
                if scalp_dir != "none":
                    return scalp_score, scalp_dir

    if price <= lower_bound:
        if not is_in_trend(recent, cfg.trend_ema_fast, cfg.trend_ema_slow, "long"):
            return 0.0, "none"
        distance = centre - price
        score = min(distance / half_range, 1.0)
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = MODEL.predict(df)
                score = (score + ml_score) / 2
            except Exception:
                pass
        if cfg.atr_normalization:
            score = normalize_score_by_volatility(df, score)
        logger.info("Signal for %s: %s, %s", symbol, score, "long")
        return score, "long"

    if price >= upper_bound:
        if not is_in_trend(recent, cfg.trend_ema_fast, cfg.trend_ema_slow, "short"):
            return 0.0, "none"
        distance = price - centre
        score = min(distance / half_range, 1.0)
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = MODEL.predict(df)
                score = (score + ml_score) / 2
            except Exception:
                pass
        if cfg.atr_normalization:
            score = normalize_score_by_volatility(df, score)
        logger.info("Signal for %s: %s, %s", symbol, score, "short")
        return score, "short"

    logger.info("Signal for %s: %s, %s", symbol, 0.0, "none")
    return 0.0, "none"


class regime_filter:
    """Match sideways regime for grid trading."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "sideways"


class Strategy:
    """Strategy wrapper so :func:`load_strategies` can auto-register it."""

    def __init__(self) -> None:
        self.name = "grid_bot"
        self.generate_signal = generate_signal
        self.regime_filter = regime_filter


__all__ = ["generate_signal", "regime_filter", "Strategy"]
