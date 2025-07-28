"""Reinforcement learning helpers."""

from . import strategy_selector
try:  # pragma: no cover - optional dependencies
    from .rl import RLStrategySelector, train_rl_selector, get_rl_strategy
except ImportError as exc:  # pragma: no cover - fallback when deps missing
    RLStrategySelector = None

    def train_rl_selector(*_a, **_k):
        raise NotImplementedError("RL dependencies missing") from exc

    def get_rl_strategy(*_a, **_k):
        raise NotImplementedError("RL dependencies missing") from exc

__all__ = ["strategy_selector"]
if RLStrategySelector is not None:
    __all__ += ["RLStrategySelector", "train_rl_selector", "get_rl_strategy"]
