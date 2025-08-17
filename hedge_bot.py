"""Hedge bot using VaR and Kelly position sizing.

This module provides a thin wrapper around a ``CoinTraderTrainer`` that is
responsible for computing Value at Risk (VaR) from historical drawdowns. The
bot uses the Kelly criterion to determine hedge size and places hedge orders on
Kraken futures.  It is intentionally lightweight so it can be integrated into
existing trading pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import pandas as pd

from crypto_bot.risk.risk_manager import kelly_fraction


class Trainer(Protocol):
    """Protocol describing the trainer interface used for VaR estimates."""

    def drawdown_series(self, lookback: int) -> Iterable[float]:
        """Return a sequence of historical drawdowns for ``lookback`` days."""


class KrakenClient(Protocol):
    """Minimal Kraken client interface expected by :class:`HedgeBot`."""

    def create_order(self, symbol: str, side: str, amount: float) -> None:
        """Place an order on the Kraken futures exchange."""


@dataclass
class HedgeConfig:
    """Runtime configuration for :class:`HedgeBot`."""

    lookback_days: int = 365
    var_confidence: float = 0.95
    futures_symbol: str = "PI_XBTUSD"
    win_prob: float = 0.5
    win_loss_ratio: float = 1.0


class HedgeBot:
    """Simple hedging helper using VaR and the Kelly criterion."""

    def __init__(self, trainer: Trainer, client: KrakenClient, config: HedgeConfig | None = None) -> None:
        self.trainer = trainer
        self.client = client
        self.config = config or HedgeConfig()

    def _value_at_risk(self) -> float:
        """Return the Value at Risk estimated by the trainer."""

        series = pd.Series(self.trainer.drawdown_series(self.config.lookback_days))
        if series.empty:
            return 0.0
        return float(series.quantile(1 - self.config.var_confidence))

    def _kelly_size(self, balance: float) -> float:
        """Return hedge size based on the Kelly criterion."""

        fraction = kelly_fraction(self.config.win_prob, self.config.win_loss_ratio)
        return max(fraction, 0.0) * balance

    def hedge(self, balance: float) -> float:
        """Calculate hedge size and place a corresponding order.

        Parameters
        ----------
        balance:
            Current account balance to size the hedge from.

        Returns
        -------
        float
            The notional size of the hedge order that was submitted.
        """

        _ = self._value_at_risk()  # Currently unused but reserved for future logic
        size = self._kelly_size(balance)
        if size > 0:
            self.client.create_order(self.config.futures_symbol, "sell", size)
        return size
