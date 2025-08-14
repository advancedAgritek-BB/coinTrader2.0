"""Lightweight HFT engine used for unit testing.

The real project likely provides a much more sophisticated implementation.
For the purposes of these exercises the engine simply stores which symbols
have been attached along with the strategy callable that should handle them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "hft.log")


@dataclass
class HFTEngine:
    """Minimal stub representing a high-frequency trading engine."""

    strategies: Dict[str, Callable] = field(default_factory=dict)

    def attach(self, symbol: str, strategy: Callable) -> None:
        """Attach ``strategy`` to ``symbol``.

        Parameters
        ----------
        symbol:
            Trading pair that should be handled by the HFT engine.
        strategy:
            Callable implementing the trading logic.
        """

        logger.info("HFTEngine attaching %s to %s", symbol, getattr(strategy, "__name__", str(strategy)))
        self.strategies[symbol] = strategy
