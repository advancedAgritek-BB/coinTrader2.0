from dataclasses import dataclass
import dataclasses
from typing import Any, Optional, Mapping

import pandas as pd
from math import isnan

from crypto_bot.capital_tracker import CapitalTracker

from crypto_bot.sentiment_filter import boost_factor, too_bearish
from crypto_bot.volatility_filter import too_flat, too_hot, calc_atr

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils import trade_memory
from crypto_bot.utils import ev_tracker
from crypto_bot.utils.strategy_utils import compute_drawdown


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
    volume_threshold_ratio: float = 0.1
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
    ) -> float:
        """Return the trade value for a signal.

        When ``stop_distance`` or ``atr`` is provided the size is calculated
        using ``risk_pct`` relative to that distance.  Otherwise the fixed
        ``trade_size_pct`` is scaled by volatility and current drawdown.
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

        logger.info(
            "Calculated position size: %.4f (vol %.2f risk %.2f)",
            size,
            volatility_factor,
            capital_risk_factor,
        )
        return size

    def allow_trade(self, df: Any, strategy: str | None = None) -> tuple[bool, str]:
        """Assess whether market conditions merit taking a trade.

        Parameters
        ----------
        df : Any
            DataFrame of OHLCV data.

        Returns
        -------
        tuple[bool, str]
            ``True``/``False`` along with the reason for the decision.
        """
        df_len = len(df)
        logger.info("[EVAL] Data length: %d", df_len)

        if df_len < 20:
            reason = "Not enough data to trade"
            logger.info("[EVAL] %s (len=%d)", reason, df_len)
            return False, reason

        if self.config.symbol and trade_memory.should_avoid(self.config.symbol):
            reason = "Symbol blocked by trade memory"
            logger.info("[EVAL] %s", reason)
            return False, reason

        if too_bearish(self.config.min_fng, self.config.min_sentiment):
            reason = "Sentiment too bearish"
            logger.info("[EVAL] %s", reason)
            return False, reason

        if too_flat(df, self.config.min_atr_pct):
            reason = "Market volatility too low"
            logger.info("[EVAL] %s", reason)
            return False, reason

        if self.config.symbol and too_hot(self.config.symbol, self.config.max_funding_rate):
            reason = "Funding rate too high"
            logger.info("[EVAL] %s", reason)
            return False, reason

        last_close = df["close"].iloc[-1]
        last_time = str(df.index[-1])
        logger.info(
            f"{self.config.symbol} | Last close: {last_close:.2f}, Time: {last_time}"
        )

        vol_mean = df["volume"].rolling(20).mean().iloc[-1]
        current_volume = df["volume"].iloc[-1]
        vol_threshold = vol_mean * self.config.volume_threshold_ratio
        logger.info(
            (
                "[EVAL] len=%d volume=%.4f (mean %.4f | min %.4f | threshold %.4f)"
            ),
            df_len,
            current_volume,
            vol_mean,
            self.config.min_volume,
            vol_threshold,
        )

        # Volume checks using configured thresholds
        if current_volume < self.config.min_volume:
            reason = "Volume < min volume threshold"
            logger.info(
                "[EVAL] %s (%.2f < %.2f)",
                reason,
                current_volume,
                self.config.min_volume,
            )
            return False, reason

        if current_volume < vol_threshold:
            percent = self.config.volume_threshold_ratio * 100
            reason = f"Volume < {percent:.0f}% of mean volume"
            logger.info(
                "[EVAL] %s (%.2f < %.2f)",
                reason,
                current_volume,
                vol_threshold,
            )
            return False, reason
        vol_std = df["close"].rolling(20).std().iloc[-1]
        prev_period_std = (
            df["close"].iloc[-21:-1].std() if len(df) >= 21 else float("nan")
        )
        if not isnan(prev_period_std) and vol_std < prev_period_std * 0.5:
            reason = "Volatility too low"
            logger.info(
                "[EVAL] %s (%.4f < %.4f)",
                reason,
                vol_std,
                prev_period_std * 0.5,
            )
            return False, reason

        if strategy is not None:
            ev = ev_tracker.get_expected_value(strategy)
            if ev == 0.0:
                stats = ev_tracker._load_stats().get(strategy, {})
                if not stats:
                    if self.config.default_expected_value is not None:
                        ev = self.config.default_expected_value
                    else:
                        ev = None
            if ev is not None and ev < self.config.min_expected_value:
                reason = (
                    f"Expected value {ev:.4f} below {self.config.min_expected_value}"
                )
                logger.info("[EVAL] %s", reason)
                return False, reason

        drawdown = compute_drawdown(
            df, lookback=self.config.pair_drawdown_lookback
        )
        if (
            self.config.max_pair_drawdown > 0
            and abs(drawdown) > self.config.max_pair_drawdown
        ):
            reason = (
                f"Pair drawdown {abs(drawdown):.2f} exceeds {self.config.max_pair_drawdown}"
            )
            logger.info("[EVAL] %s", reason)
            return False, reason

        self.boost = boost_factor(self.config.bull_fng, self.config.bull_sentiment)
        logger.info(
            f"[EVAL] Trade allowed for {self.config.symbol} – Volume {current_volume:.4f} >= {self.config.volume_threshold_ratio*100}% of mean {vol_mean:.4f}"
        )
        reason = f"Trade allowed (boost {self.boost:.2f})"
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
