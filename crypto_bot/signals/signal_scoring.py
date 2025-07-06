from typing import Tuple, Callable, Optional, Iterable, Dict
import pandas as pd
import asyncio
from crypto_bot.ml_signal_model import predict_signal
from crypto_bot.indicators.cycle_bias import get_cycle_bias
from crypto_bot.utils.strategy_utils import compute_drawdown
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


def evaluate(
    strategy_fn: Callable[[pd.DataFrame], Tuple],
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str, Optional[float]]:
    """Evaluate signal from a strategy callable."""
    if config is not None:
        try:
            result = strategy_fn(df, config)
        except TypeError:
            result = strategy_fn(df)
    else:
        result = strategy_fn(df)

    if isinstance(result, tuple):
        score, direction, *extras = result
        atr = extras[0] if extras else None
    else:
        score, direction = result, "none"
        atr = None
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

    return score, direction, atr


async def evaluate_async(
    strategy_fn: Callable[[pd.DataFrame], Tuple],
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[float, str, Optional[float]]:
    """Asynchronous wrapper around ``evaluate``."""
    return await asyncio.to_thread(evaluate, strategy_fn, df, config)


def evaluate_strategies(
    strategies: Iterable[Callable[[pd.DataFrame], Tuple]],
    df: pd.DataFrame,
    config: Optional[Dict] = None,
) -> Dict[str, object]:
    """Return best scoring strategy evaluation.

    Each strategy is evaluated and combined with simple sharpe-like metrics and
    drawdown. Any strategy raising an exception is skipped and logged.
    """

    best_score = float("-inf")
    best_res: Dict[str, object] = {"score": 0.0, "direction": "none", "name": ""}
    rets = df["close"].pct_change().dropna()
    sharpe = 0.0
    if len(rets) > 1 and rets.std() != 0:
        sharpe = float(rets.mean() / rets.std() * (len(rets) ** 0.5))
    drawdown = compute_drawdown(df)

    for strat in strategies:
        try:
            score, direction, _ = evaluate(strat, df, config)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Strategy %s failed: %s", getattr(strat, "__name__", str(strat)), exc)
            continue

        metric = score + sharpe + drawdown
        if metric > best_score:
            best_score = metric
            best_res = {"score": score, "direction": direction, "name": getattr(strat, "__name__", "")}

    return best_res
