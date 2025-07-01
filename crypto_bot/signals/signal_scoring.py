from typing import Tuple, Callable
import pandas as pd
import asyncio


def evaluate(strategy_fn: Callable[[pd.DataFrame], Tuple[float, str]], df: pd.DataFrame) -> Tuple[float, str]:
    """Evaluate signal from a strategy callable."""
    score, direction = strategy_fn(df)
    score = max(0.0, min(score, 1.0))
    return score, direction


async def evaluate_async(
    strategy_fn: Callable[[pd.DataFrame], Tuple[float, str]],
    df: pd.DataFrame,
) -> Tuple[float, str]:
    """Asynchronous wrapper around ``evaluate``."""
    return await asyncio.to_thread(evaluate, strategy_fn, df)
