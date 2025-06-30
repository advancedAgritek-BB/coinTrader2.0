from typing import Tuple, Callable
import pandas as pd


def evaluate(strategy_fn: Callable[[pd.DataFrame], Tuple[float, str]], df: pd.DataFrame) -> Tuple[float, str]:
    """Evaluate signal from a strategy callable."""
    score, direction = strategy_fn(df)
    score = max(0.0, min(score, 1.0))
    return score, direction
