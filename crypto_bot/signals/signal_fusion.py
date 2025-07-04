from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import pandas as pd

from .signal_scoring import evaluate


class SignalFusionEngine:
    """Combine signals from multiple strategies using weighted averaging."""

    def __init__(self, strategies: Iterable[Tuple[Callable[[pd.DataFrame], Tuple[float, str]], float]]):
        self.strategies: List[Tuple[Callable[[pd.DataFrame], Tuple[float, str]], float]] = list(strategies)

    def fuse(self, df: pd.DataFrame, config: dict | None = None) -> Tuple[float, str]:
        if not self.strategies:
            return 0.0, "none"

        total_weight = 0.0
        weighted_score = 0.0
        long_votes = 0
        short_votes = 0
        signed_sum = 0.0

        for fn, weight in self.strategies:
            score, direction = evaluate(fn, df, config)
            weighted_score += score * weight
            total_weight += weight

            if direction == "long":
                long_votes += 1
                signed_sum += score * weight
            elif direction == "short":
                short_votes += 1
                signed_sum -= score * weight

        score = weighted_score / total_weight if total_weight else 0.0

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

