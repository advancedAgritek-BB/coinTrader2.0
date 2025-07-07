from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class BacktestConfig:
    """Configuration for running backtests."""

    symbol: str
    timeframe: str
    since: int
    limit: int
    mode: str
    stop_loss_range: Iterable[float]
    take_profit_range: Iterable[float]
    risk_per_trade_pct: float = 0.01
    trailing_stop_atr_mult: float = 1.0
    partial_tp_atr_mult: float = 0.5
    maker_fee_pct: float = 0.0002
    taker_fee_pct: float = 0.0007
    slippage_base_pct: float = 0.001
    slippage_vol_factor: float = 0.01
    seed: int | None = None


__all__ = ["BacktestConfig"]
