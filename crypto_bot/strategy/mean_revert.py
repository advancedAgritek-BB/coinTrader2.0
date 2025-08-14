import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import ta

from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "bot.log")


@dataclass
class Position:
    side: str  # "long" or "short"
    entry_price: float
    entry_bar: int
    stop: float


@dataclass
class Config:
    ema_len: int = 20
    adx_max: float = 25.0
    z_entry: float = 1.0
    z_exit: float = 0.5
    atr_stop_mult: float = 1.5
    max_spread_bp: float = 5.0
    time_exit_bars: int = 30
    adx_window: int = 14
    atr_period: int = 14
    std_len: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Config":
        data = data or {}
        return cls(**{f.name: data.get(f.name, getattr(cls, f.name)) for f in dataclasses.fields(cls)})


def generate_signal(
    df: pd.DataFrame,
    position: Optional[Position] = None,
    config: Optional[dict] = None,
    *,
    spread_bp: float = 0.0,
) -> Tuple[float, str, Optional[float], Optional[Position]]:
    """Mean-reversion strategy with ATR stop and time/z exits.

    Returns a tuple ``(score, action, stop, new_position)`` where ``action`` is
    one of ``"long"``, ``"short"``, ``"exit"`` or ``"none"``.  ``stop`` is the
    suggested stop loss for new entries.  ``new_position`` describes the
    position after taking the action and should be persisted by the caller.
    """

    if df is None or df.empty:
        return 0.0, "none", None, position

    cfg = Config.from_dict(config)

    std_len = cfg.std_len or cfg.ema_len
    lookback = max(cfg.ema_len, std_len, cfg.adx_window, cfg.atr_period) + 1
    if len(df) < lookback:
        return 0.0, "none", None, position

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema = ta.trend.ema_indicator(close, window=cfg.ema_len)
    std = close.rolling(std_len).std()
    adx = ta.trend.ADXIndicator(high, low, close, window=cfg.adx_window).adx()
    atr = ta.volatility.average_true_range(high, low, close, window=cfg.atr_period)

    price = close.iloc[-1]
    ema_v = ema.iloc[-1]
    std_v = std.iloc[-1]
    adx_v = adx.iloc[-1]
    atr_v = atr.iloc[-1]
    z = (price - ema_v) / std_v if std_v > 0 else 0.0

    if position is None:
        if adx_v >= cfg.adx_max or spread_bp > cfg.max_spread_bp:
            return 0.0, "none", None, None
        if z <= -cfg.z_entry:
            stop = price - cfg.atr_stop_mult * atr_v
            logger.info("mr_entry side=long price=%s z=%s", price, z)
            pos = Position("long", price, len(df) - 1, stop)
            return 1.0, "long", stop, pos
        if z >= cfg.z_entry:
            stop = price + cfg.atr_stop_mult * atr_v
            logger.info("mr_entry side=short price=%s z=%s", price, z)
            pos = Position("short", price, len(df) - 1, stop)
            return 1.0, "short", stop, pos
        return 0.0, "none", None, None

    bars_held = len(df) - position.entry_bar
    action = "none"
    stop = None
    new_pos = position

    if position.side == "long":
        if price <= position.stop:
            logger.info("mr_stop side=long price=%s stop=%s", price, position.stop)
            return 1.0, "exit", None, None
        if z >= -cfg.z_exit:
            logger.info("mr_exit_z side=long price=%s z=%s", price, z)
            return 1.0, "exit", None, None
        if bars_held >= cfg.time_exit_bars:
            logger.info("mr_exit_time side=long price=%s bars=%s", price, bars_held)
            return 1.0, "exit", None, None
    else:  # short
        if price >= position.stop:
            logger.info("mr_stop side=short price=%s stop=%s", price, position.stop)
            return 1.0, "exit", None, None
        if z <= cfg.z_exit:
            logger.info("mr_exit_z side=short price=%s z=%s", price, z)
            return 1.0, "exit", None, None
        if bars_held >= cfg.time_exit_bars:
            logger.info("mr_exit_time side=short price=%s bars=%s", price, bars_held)
            return 1.0, "exit", None, None

    return 0.0, action, stop, new_pos


class regime_filter:
    """Match mean-reverting regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "mean-reverting"
