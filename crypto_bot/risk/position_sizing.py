"""Volatility-aware position sizing and basic risk management helpers."""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import numpy as np

from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "risk.log")

# store recent price history per symbol for volatility estimates
_PRICE_HISTORY: Dict[str, Deque[Tuple[float, float]]] = defaultdict(deque)


def record_price(symbol: str, price: float, timestamp: float | None = None) -> None:
    """Record ``price`` for ``symbol`` at ``timestamp``.

    Parameters
    ----------
    symbol:
        Market symbol being tracked.
    price:
        Trade price.
    timestamp:
        UNIX timestamp in seconds.  Defaults to ``time.time()``.
    """

    ts = time.time() if timestamp is None else float(timestamp)
    _PRICE_HISTORY[symbol].append((ts, float(price)))


def compute_realized_vol(symbol: str, horizon_secs: int) -> float:
    """Return the per-second volatility over the last ``horizon_secs`` seconds.

    The function calculates the standard deviation of log returns of all recorded
    prices for ``symbol`` within the lookback window.  If fewer than two prices
    are available the function returns ``0.0``.
    """

    if horizon_secs <= 0:
        return 0.0

    cutoff = time.time() - horizon_secs
    hist = _PRICE_HISTORY.get(symbol)
    if not hist:
        return 0.0

    # drop stale data
    while hist and hist[0][0] < cutoff:
        hist.popleft()

    prices = [p for _, p in hist]
    if len(prices) < 2:
        return 0.0

    returns = np.diff(np.log(prices))
    if returns.size == 0:
        return 0.0
    return float(np.std(returns, ddof=1))


def size_for_sigma(target_sigma_usd: float, vol: float, price: float, min_notional: float = 0.0) -> float:
    """Return base position size so that dollar volatility ≈ ``target_sigma_usd``.

    ``vol`` is interpreted as the per-second volatility of log returns.  The
    resulting size is floored to ``min_notional``.
    """

    if price <= 0 or vol <= 0:
        return 0.0
    size = target_sigma_usd / (price * vol)
    if min_notional > 0:
        size = math.floor(size / min_notional) * min_notional
    return max(size, 0.0)


@dataclass
class RiskLimits:
    """Configuration for the :class:`RiskManager`."""

    starting_equity: float
    daily_loss_limit_pct: float
    max_consecutive_losses: int = 3
    symbol_cooldown_min: float = 60.0


class RiskManager:
    """Simple risk manager implementing daily loss halt and symbol cooldowns."""

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.daily_pnl = 0.0
        self.day_start = time.gmtime().tm_yday
        self.halt = False
        self.consecutive_losses: Dict[str, int] = defaultdict(int)
        self.cooldowns: Dict[str, float] = {}

    def _reset_daily_if_needed(self) -> None:
        today = time.gmtime().tm_yday
        if today != self.day_start:
            self.day_start = today
            self.daily_pnl = 0.0
            self.halt = False
            self.cooldowns.clear()
            self.consecutive_losses.clear()

    def allow_entry(self, symbol: str) -> bool:
        """Return ``True`` if a new trade for ``symbol`` is allowed."""

        self._reset_daily_if_needed()
        now = time.time()
        if self.halt:
            return False
        expiry = self.cooldowns.get(symbol)
        if expiry and now < expiry:
            return False
        if expiry and now >= expiry:
            self.cooldowns.pop(symbol, None)
            self.consecutive_losses[symbol] = 0
        return True

    def record_result(self, symbol: str, pnl: float) -> None:
        """Record realized ``pnl`` for ``symbol`` and update risk state."""

        self._reset_daily_if_needed()
        self.daily_pnl += pnl
        loss_limit = -abs(self.limits.daily_loss_limit_pct) * self.limits.starting_equity
        if self.daily_pnl <= loss_limit:
            if not self.halt:
                logger.warning(
                    "risk_halt: daily PnL %.2f breached limit %.2f",
                    self.daily_pnl,
                    loss_limit,
                )
            self.halt = True

        if pnl < 0:
            self.consecutive_losses[symbol] += 1
            if self.consecutive_losses[symbol] >= self.limits.max_consecutive_losses:
                cooldown = time.time() + self.limits.symbol_cooldown_min * 60
                self.cooldowns[symbol] = cooldown
                self.consecutive_losses[symbol] = 0
        else:
            self.consecutive_losses[symbol] = 0
