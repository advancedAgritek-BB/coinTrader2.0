from dataclasses import dataclass
from typing import Any

from crypto_bot.capital_tracker import CapitalTracker

from crypto_bot.sentiment_filter import boost_factor, too_bearish
from crypto_bot.volatility_filter import too_flat, too_hot

from crypto_bot.utils.logger import setup_logger

# Log to the main bot file so risk messages are consolidated
logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


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
    strategy_allocation: dict | None = None
    volume_ratio: float = 1.0


class RiskManager:
    """Utility class for evaluating account and trade level risk."""

    def __init__(self, config: RiskConfig) -> None:
        """Create a new manager with the given risk configuration."""
        self.config = config
        self.capital_tracker = CapitalTracker(config.strategy_allocation or {})
        self.equity = 1.0
        self.peak_equity = 1.0
        self.stop_order = None
        self.boost = 1.0

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

    def position_size(self, confidence: float, balance: float) -> float:
        """Calculate position size based on signal confidence and balance."""
        size = balance * confidence * self.config.trade_size_pct
        logger.info("Calculated position size: %.4f", size)
        return size

    def allow_trade(self, df: Any) -> tuple[bool, str]:
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
        if len(df) < 20:
            reason = "Not enough data to trade"
            logger.info(reason)
            return False, reason

        if too_bearish(self.config.min_fng, self.config.min_sentiment):
            reason = "Sentiment too bearish"
            logger.info(reason)
            return False, reason

        if too_flat(df, self.config.min_atr_pct):
            reason = "Market volatility too low"
            logger.info(reason)
            return False, reason

        if self.config.symbol and too_hot(self.config.symbol, self.config.max_funding_rate):
            reason = "Funding rate too high"
            logger.info(reason)
            return False, reason

        vol_mean = df['volume'].rolling(20).mean().iloc[-1]
        if df['volume'].iloc[-1] < vol_mean * 0.5:
            reason = f"Volume {df['volume'].iloc[-1]:.4f} below mean {vol_mean:.4f}"
            logger.info(reason)
            return False, reason
        vol_std = df['close'].rolling(20).std().iloc[-1]
        if vol_std < df['close'].iloc[-20:-1].std() * 0.5:
            reason = "Volatility too low"
            logger.info(reason)
            return False, reason
        self.boost = boost_factor(self.config.bull_fng, self.config.bull_sentiment)
        reason = f"Trade allowed (boost {self.boost:.2f})"
        logger.info(reason)
        return True, reason

    def register_stop_order(self, order: dict) -> None:
        """Store the protective stop order."""
        self.stop_order = order
        logger.info("Registered stop order %s", order)

    def update_stop_order(self, new_amount: float) -> None:
        """Update the stored stop order amount."""
        if self.stop_order:
            self.stop_order["amount"] = new_amount
            logger.info("Updated stop order amount to %.4f", new_amount)

    def cancel_stop_order(self, exchange) -> None:
        """Cancel the existing stop order on the exchange if needed."""
        if not self.stop_order:
            return
        order = self.stop_order
        if not order.get("dry_run") and "id" in order:
            try:
                exchange.cancel_order(order["id"], order.get("symbol"))
                logger.info("Cancelled stop order %s", order.get("id"))
            except Exception as e:
                logger.error("Failed to cancel stop order: %s", e)
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
