from typing import Tuple, Callable, Optional
import pandas as pd
import asyncio
from crypto_bot.ml_signal_model import predict_signal
from crypto_bot.indicators.cycle_bias import get_cycle_bias


def evaluate(
    strategy_fn: Callable[[pd.DataFrame], Tuple[float, str]],
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str]:
    """Evaluate signal from a strategy callable."""
    if config is not None:
        try:
            score, direction = strategy_fn(df, config)
        except TypeError:
            score, direction = strategy_fn(df)
    else:
        score, direction = strategy_fn(df)
    score = max(0.0, min(score, 1.0))

    if config:
        ml_cfg = config.get("ml_signal_model", {})
        if ml_cfg.get("enabled"):
            weight = ml_cfg.get("weight", 0.5)
            try:
                ml_score = predict_signal(df)
                score = (score * (1 - weight)) + (ml_score * weight)
                score = max(0.0, min(score, 1.0))
            except Exception:
                pass

        bias_cfg = config.get("cycle_bias", {})
        if bias_cfg.get("enabled"):
            try:
                bias = get_cycle_bias(bias_cfg)
                score *= bias
                score = max(0.0, min(score, 1.0))
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
