from dataclasses import dataclass
import dataclasses
from typing import Any, Optional, Mapping

import pandas as pd
from math import isnan

from crypto_bot.capital_tracker import CapitalTracker

from crypto_bot.volatility_filter import too_flat, calc_atr

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.regime_pnl_tracker import get_recent_win_rate


# Log to the main bot file so risk messages are consolidated
logger = setup_logger(__name__, LOG_DIR / "bot.log")


@dataclass
class RiskConfig:
    """Configuration values governing risk limits."""

    max_drawdown: float
    stop_loss_pct: float
    take_profit_pct: float
    min_fng: int = 0
    min_sentiment: int = 0
    bull_fng: int = 101
    bull_sentiment: int = 101
    min_atr_pct: float = 0.0
    max_funding_rate: float = 1.0
    symbol: str = ""
    trade_size_pct: float = 0.1
    risk_pct: float = 0.01
    min_volume: float = 0.0
    volume_threshold_ratio: float = 0.05
    strategy_allocation: dict | None = None
    volume_ratio: float = 1.0
    atr_short_window: int = 14
    atr_long_window: int = 50
    max_volatility_factor: float = 1.5
    min_expected_value: float = 0.0
    default_expected_value: float | None = None
    atr_period: int = 14
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 4.0
    max_pair_drawdown: float = 0.0
    pair_drawdown_lookback: int = 20
    min_history_bars: int = 20
    win_rate_threshold: float = 0.7
    win_rate_boost_factor: float = 1.5


class RiskManager:
    """Utility class for evaluating account and trade level risk."""

    def __init__(self, config: RiskConfig) -> None:
        """Create a new manager with the given risk configuration."""
        self.config = config
        self.capital_tracker = CapitalTracker(config.strategy_allocation or {})
        self.equity = 1.0
        self.peak_equity = 1.0
        self.stop_orders: dict[str, dict] = {}
        self.stop_order: dict | None = None
        # Track protective stop orders for each open trade by symbol
        self.boost = 1.0

    @classmethod
    def from_config(cls, cfg: Mapping) -> "RiskManager":
        """Instantiate ``RiskManager`` from a configuration mapping.

        Parameters
        ----------
        cfg : Mapping
            Dictionary with keys corresponding to :class:`RiskConfig` fields.

        Returns
        -------
        RiskManager
            Newly created instance using the provided configuration.
        """
        params = {}
        for f in dataclasses.fields(RiskConfig):
            if f.default is not dataclasses.MISSING:
                default = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
                default = f.default_factory()
            else:
                default = None
            params[f.name] = cfg.get(f.name, default)
        config = RiskConfig(**params)
        return cls(config)

    def get_stop_order(self, symbol: str) -> dict | None:
        """Return the stop order for ``symbol`` if present."""
        return self.stop_orders.get(symbol)

    def update_allocation(self, weights: dict) -> None:
        """Update strategy allocation weights at runtime."""
        self.config.strategy_allocation = weights
        self.capital_tracker.update_allocation(weights)

    def update_equity(self, new_equity: float) -> bool:
        """Update current equity and evaluate drawdown limit.

        Parameters
        ----------
        new_equity : float
            The account equity after the most recent trade.

        Returns
        -------
        bool
            ``True`` if drawdown remains under ``max_drawdown``.
        """
        self.equity = new_equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        drawdown = 1 - self.equity / self.peak_equity
        logger.info(
            "Equity updated to %.2f (drawdown %.2f)",
            self.equity,
            drawdown,
        )
        return drawdown < self.config.max_drawdown

    def position_size(
        self,
        confidence: float,
        balance: float,
        df: Optional[pd.DataFrame] = None,
        stop_distance: float | None = None,
        atr: float | None = None,
        price: float | None = None,
        name: str | None = None,
    ) -> float:
        """Return the trade value for a signal.

        When ``stop_distance`` or ``atr`` is provided the size is calculated
        using ``risk_pct`` relative to that distance.  Otherwise the fixed
        ``trade_size_pct`` is scaled by volatility and current drawdown.
        When ``name`` is supplied the recent win rate for that strategy is
        fetched and the size is boosted by ``win_rate_boost_factor`` when the
        rate exceeds ``win_rate_threshold``.
        """

        volatility_factor = 1.0
        if df is not None and not df.empty:
            short_atr = calc_atr(df, window=self.config.atr_short_window)
            long_atr = calc_atr(df, window=self.config.atr_long_window)
            if long_atr > 0 and not isnan(short_atr) and not isnan(long_atr):
                volatility_factor = min(
                    short_atr / long_atr,
                    self.config.max_volatility_factor,
                )

        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = 1 - self.equity / self.peak_equity
        if self.config.max_drawdown > 0:
            capital_risk_factor = max(
                0.0, 1 - drawdown / self.config.max_drawdown
            )
        else:
            capital_risk_factor = 1.0

        if stop_distance is not None or atr is not None:
            risk_value = balance * self.config.risk_pct * confidence
            stop_loss_distance = atr if atr and atr > 0 else stop_distance
            trade_price = price or 1.0
            if stop_loss_distance and stop_loss_distance > 0:
                size = risk_value * trade_price / stop_loss_distance
            else:
                size = balance * confidence * self.config.trade_size_pct
            max_size = balance * self.config.trade_size_pct
            if size > max_size:
                size = max_size
        else:
            size = (
                balance
                * self.config.trade_size_pct
                * confidence
                * volatility_factor
                * capital_risk_factor
            )

        if name:
            try:
                win_rate = get_recent_win_rate(strategy=name)
            except Exception:
                win_rate = 0.0
            if win_rate > self.config.win_rate_threshold:
                size *= self.config.win_rate_boost_factor

        if size <= 0:
            logger.info(
                "Position size zero - balance=%.2f trade_size_pct=%.2f",
                balance,
                self.config.trade_size_pct,
            )

        logger.info(
            "Calculated position size: %.4f (vol %.2f risk %.2f)",
            size,
            volatility_factor,
            capital_risk_factor,
        )
        return size

    def allow_trade(
        self,
        df: Any,
        strategy: str | None = None,
        symbol: str | None = None,
        score: float | None = None,
    ) -> tuple[bool, str]:
        """Assess whether market conditions merit taking a trade.

        Parameters
        ----------
        df : Any
            DataFrame of OHLCV data.
        symbol : str, optional
            Trading pair being evaluated. Defaults to ``self.config.symbol``.

        Returns
        -------
        tuple[bool, str]
            ``True``/``False`` along with the reason for the decision.
        """
        df_len = len(df)
        logger.info("[EVAL] Data length: %d", df_len)

        if df_len < self.config.min_history_bars:
            reason = "Not enough data to trade"
            logger.info("[EVAL] %s (len=%d)", reason, df_len)
            return False, reason

        current_volume = df["volume"].iloc[-1]

        if score is not None and score > 0.4:
            reason = "High score boost"
            logger.info("[EVAL] %s", reason)
            return True, reason

        if current_volume < 0.001:
            reason = "Volume too low"
            logger.info("[EVAL] %s", reason)
            return False, reason

        if too_flat(df, 0.00001):
            reason = "Volatility too low for HFT"
            logger.info("[EVAL] %s", reason)
            return False, reason

        if too_flat(df, 0.00005):
            reason = "Volatility too low"
            logger.info("[EVAL] %s", reason)
            return False, reason

        self.boost = 1.0
        reason = "Trade allowed"
        logger.info(
            "Allow %s: vol=%.6f, flat=%s",
            symbol,
            current_volume,
            too_flat(df, 0.00005),
        )
        logger.info("[EVAL] %s", reason)
        return True, reason

    def register_stop_order(
        self,
        order: dict,
        strategy: str | None = None,
        symbol: str | None = None,
        entry_price: float | None = None,
        confidence: float | None = None,
        direction: str | None = None,
        take_profit: float | None = None,
        regime: str | None = None,
    ) -> None:
        """Store the protective stop order and related trade info."""
        order = dict(order)
        if strategy is not None:
            order["strategy"] = strategy
        if symbol is None:
            symbol = order.get("symbol")
        if symbol is None:
            raise ValueError("Symbol required to register stop order")
        order["symbol"] = symbol
        if entry_price is not None:
            order["entry_price"] = entry_price
        if confidence is not None:
            order["confidence"] = confidence
        if direction is not None:
            order["direction"] = direction
        if take_profit is not None:
            order["take_profit"] = take_profit
        if regime is not None:
            order["regime"] = regime
        self.stop_order = order
        if symbol is not None:
            self.stop_orders[symbol] = order
        logger.info("Registered stop order %s", order)

    def update_stop_order(self, new_amount: float, symbol: str | None = None) -> None:
        """Update the stored stop order amount."""
        order = self.stop_orders.get(symbol) if symbol else self.stop_order
        if not order:
            return
        order["amount"] = new_amount
        if symbol:
            self.stop_orders[symbol] = order
        else:
            self.stop_order = order
        logger.info("Updated stop order amount to %.4f", new_amount)

    def cancel_stop_order(self, exchange, symbol: str | None = None) -> None:
        """Cancel the existing stop order if needed."""
        order = self.stop_orders.get(symbol) if symbol else self.stop_order
        if not order:
            return
        if not order.get("dry_run") and "id" in order:
            try:
                exchange.cancel_order(order["id"], order.get("symbol"))
                logger.info("Cancelled stop order %s", order.get("id"))
            except Exception as e:
                logger.error("Failed to cancel stop order: %s", e)
        if symbol:
            self.stop_orders.pop(symbol, None)
        else:
            self.stop_order = None

    def can_allocate(self, strategy: str, amount: float, balance: float) -> bool:
        """Check if ``strategy`` can use additional ``amount`` capital."""
        return self.capital_tracker.can_allocate(strategy, amount, balance)

    def allocate_capital(self, strategy: str, amount: float) -> None:
        """Record capital allocation for a strategy."""
        self.capital_tracker.allocate(strategy, amount)

    def deallocate_capital(self, strategy: str, amount: float) -> None:
        """Release previously allocated capital."""
        self.capital_tracker.deallocate(strategy, amount)
