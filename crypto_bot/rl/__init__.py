"""Reinforcement learning helpers."""

from . import strategy_selector
try:  # pragma: no cover - optional dependencies
    from .rl import RLStrategySelector, train_rl_selector, get_rl_strategy
except Exception:  # pragma: no cover - fallback when deps missing
    RLStrategySelector = None

    def train_rl_selector(*_a, **_k):
        raise NotImplementedError("RL dependencies missing")

    def get_rl_strategy(*_a, **_k):
        raise NotImplementedError("RL dependencies missing")

__all__ = [
    "strategy_selector",
    "RLStrategySelector",
    "train_rl_selector",
    "get_rl_strategy",
]
