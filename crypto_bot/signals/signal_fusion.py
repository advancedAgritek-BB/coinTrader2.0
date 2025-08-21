from __future__ import annotations

from typing import Callable, Iterable, List, Tuple, Dict

import pandas as pd

from .signal_scoring import evaluate
from .weight_optimizer import OnlineWeightOptimizer, get_optimizer


class SignalFusionEngine:
    """Combine signals from multiple strategies using weighted averaging."""

    def __init__(
        self,
        strategies: Iterable[Tuple[Callable[[pd.DataFrame], Tuple[float, str]], float]],
        weight_optimizer: OnlineWeightOptimizer | None = None,
    ) -> None:
        self.strategies: List[Tuple[Callable[[pd.DataFrame], Tuple[float, str]], float]] = list(strategies)
        self.weight_optimizer = weight_optimizer

    def fuse(self, df: pd.DataFrame, config: dict | None = None) -> Tuple[float, str]:
        if not self.strategies:
            return 0.0, "none"

        opt_weights: Dict[str, float] = {}
        if self.weight_optimizer and config:
            cfg = config.get("signal_weight_optimizer", {})
            if cfg.get("enabled"):
                self.weight_optimizer.learning_rate = cfg.get("learning_rate", 0.1)
                self.weight_optimizer.min_weight = cfg.get("min_weight", 0.0)
                self.weight_optimizer.update()
                opt_weights = self.weight_optimizer.get_weights()

        total_weight = 0.0
        weighted_score = 0.0
        long_votes = 0
        short_votes = 0
        signed_sum = 0.0

        for fn, weight in self.strategies:
            w = opt_weights.get(fn.__name__, weight)
            score, direction, _ = evaluate(fn, df, config)
            if direction == "none" and score == 0.0:
                continue

            weighted_score += score * w
            total_weight += w

            if direction == "long":
                long_votes += 1
                signed_sum += score * w
            elif direction == "short":
                short_votes += 1
                signed_sum -= score * w

        if total_weight == 0:
            return 0.0, "none"

        score = weighted_score / total_weight

        if long_votes > short_votes:
            direction = "long"
        elif short_votes > long_votes:
            direction = "short"
        else:
            if signed_sum > 0:
                direction = "long"
            elif signed_sum < 0:
                direction = "short"
            else:
                direction = "none"

        return score, direction


def fuse_signals(
    df: pd.DataFrame,
    strategies: Iterable[Tuple[Callable[[pd.DataFrame], Tuple[float, str]], float]],
    config: dict | None = None,
) -> Tuple[float, str]:
    """Convenience function to fuse signals from ``strategies``."""
    engine = SignalFusionEngine(strategies)
    return engine.fuse(df, config)

