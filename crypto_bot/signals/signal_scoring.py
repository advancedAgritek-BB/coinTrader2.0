from typing import Tuple, Callable, Optional, Iterable, Dict, List
import pandas as pd
import asyncio
from crypto_bot.ml_signal_model import predict_signal
from crypto_bot.indicators.cycle_bias import get_cycle_bias
from crypto_bot.utils.strategy_utils import compute_drawdown
from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "bot.log")


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
    strategy_fns: List[Callable[[pd.DataFrame], Tuple]],
    df: pd.DataFrame,
    config: Optional[dict] = None,
    max_parallel: int | None = None,
    max_parallel: int = 4,
) -> List[Tuple[float, str, Optional[float]]]:
    """Asynchronously evaluate strategy callables with limited concurrency.

    Parameters
    ----------
    strategy_fns : List[Callable]
        Strategy functions to evaluate.  Each callable must take a ``DataFrame``
        and optionally a configuration dictionary and return a tuple containing
        a score and direction.
    df : pd.DataFrame
        DataFrame containing market data.
    config : Optional[dict]
        Optional configuration passed through to the strategy functions.
    max_parallel : int | None
        Maximum number of strategies evaluated concurrently. ``None`` means no
        limit.
    max_parallel : int, default 4
        Maximum number of strategies evaluated concurrently.
    """

    max_parallel = None
    if config is not None:
        try:
            max_parallel = int(config.get("max_parallel", len(fns)))
        except Exception:
            max_parallel = len(fns)
    if max_parallel is None:
        max_parallel = len(fns)

    sem = asyncio.Semaphore(max_parallel)

    async def run_fn(fn):
        async with sem:
            return await asyncio.to_thread(evaluate, fn, df, config)

    tasks = [run_fn(fn) for fn in fns]
    sem = None
    if max_parallel is not None:
        if not isinstance(max_parallel, int) or max_parallel < 1:
            raise ValueError("max_parallel must be a positive integer or None")
        sem = asyncio.Semaphore(max_parallel)

    async def sem_eval(fn):
        async def _eval():
            return await asyncio.to_thread(evaluate, fn, df, config)

        if sem:
            async with sem:
                return await _eval()
        return await _eval()

    tasks = [asyncio.create_task(sem_eval(fn)) for fn in fns]
    results = await asyncio.gather(*tasks)
    return list(results)
    results: List[Tuple[float, str, Optional[float]]] = []
    for i in range(0, len(strategy_fns), max_parallel):
        batch = strategy_fns[i : i + max_parallel]
        tasks = [asyncio.to_thread(evaluate, fn, df, config) for fn in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

    return results


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
