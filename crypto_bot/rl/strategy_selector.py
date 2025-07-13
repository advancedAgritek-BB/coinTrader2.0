from __future__ import annotations

import pandas as pd
from pathlib import Path

from crypto_bot.utils.logger import LOG_DIR
from typing import Callable, Dict

from crypto_bot.strategy import (
    trend_bot,
    grid_bot,
    sniper_bot,
    dex_scalper,
    dca_bot,
    mean_bot,
    breakout_bot,
    solana_scalping,
)

# Default log file location
LOG_FILE = Path("crypto_bot/logs/strategy_pnl.csv")

# Map strategy names to generation functions
_STRATEGY_FN_MAP: Dict[str, Callable[[pd.DataFrame], tuple]] = {
    "trend_bot": trend_bot.generate_signal,
    "grid_bot": grid_bot.generate_signal,
    "sniper_bot": sniper_bot.generate_signal,
    "dex_scalper": dex_scalper.generate_signal,
    "dca_bot": dca_bot.generate_signal,
    "mean_bot": mean_bot.generate_signal,
    "breakout_bot": breakout_bot.generate_signal,
    "solana_scalping": solana_scalping.generate_signal,
}


class RLStrategySelector:
    """Contextual bandit using mean PnL weighted by trade counts."""

    def __init__(self) -> None:
        # {regime: {strategy: {"mean": float, "count": int}}}
        self.regime_scores: Dict[str, Dict[str, Dict[str, float]]] = {}

    def train(self, log_file: Path = LOG_FILE) -> None:
        """Train on historical PnL log."""
        if not Path(log_file).exists():
            return
        df = pd.read_csv(log_file)
        if {"regime", "strategy", "pnl"}.issubset(df.columns):
            grouped = df.groupby(["regime", "strategy"]).agg(
                mean=("pnl", "mean"), count=("pnl", "count")
            )
            for (regime, strat), row in grouped.iterrows():
                self.regime_scores.setdefault(regime, {})[strat] = {
                    "mean": float(row["mean"]),
                    "count": int(row["count"]),
                }

    def select(self, regime: str) -> Callable[[pd.DataFrame], tuple]:
        from ..strategy_router import strategy_for

        scores = self.regime_scores.get(regime)
        if not scores:
            return strategy_for(regime)
        best = max(
            scores.items(), key=lambda x: x[1]["mean"] * x[1]["count"]
        )[0]
        return _STRATEGY_FN_MAP.get(best, strategy_for(regime))


_selector = RLStrategySelector()


def train(log_file: Path = LOG_FILE) -> None:
    """Train the global selector."""
    _selector.train(log_file)


def select_strategy(regime: str) -> Callable[[pd.DataFrame], tuple]:
    """Return strategy for regime using the trained selector."""
    if not _selector.regime_scores:
        _selector.train()
    return _selector.select(regime)
