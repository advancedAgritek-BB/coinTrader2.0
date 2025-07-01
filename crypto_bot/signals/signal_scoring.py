from typing import Tuple, Callable, Optional
import pandas as pd
import asyncio
from crypto_bot.ml_signal_model import predict_signal


def evaluate(
    strategy_fn: Callable[[pd.DataFrame], Tuple[float, str]],
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str]:
    """Evaluate signal from a strategy callable."""
    score, direction = strategy_fn(df)
    score = max(0.0, min(score, 1.0))
    if config and config.get("ml_signal_model", {}).get("enabled"):
        try:
            ml_score = predict_signal(df)
            score = (score + ml_score) / 2
        except Exception:
            pass
    return score, direction


async def evaluate_async(
    strategy_fn: Callable[[pd.DataFrame], Tuple[float, str]],
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str]:
    """Asynchronous wrapper around ``evaluate``."""
    return await asyncio.to_thread(evaluate, strategy_fn, df, config)
