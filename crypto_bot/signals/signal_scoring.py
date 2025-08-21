from typing import Tuple, Callable, Optional, Iterable, Dict, List
import pandas as pd
import numpy as np
import asyncio
from crypto_bot.ml_signal_model import predict_signal
from crypto_bot.indicators.cycle_bias import get_cycle_bias
from crypto_bot.utils.strategy_utils import compute_drawdown
from crypto_bot.utils.logger import LOG_DIR, setup_logger, indicator_logger


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
        raw_score = float(result[0])
        if len(result) > 1:
            direction = result[1]
        else:
            direction = (
                "long" if raw_score > 0 else "short" if raw_score < 0 else "none"
            )
        extras = result[2:]
        atr = extras[0] if extras else None
    else:
        raw_score = float(result)
        direction = "long" if raw_score > 0 else "short" if raw_score < 0 else "none"
        atr = None
    score = max(0.0, min(abs(raw_score), 1.0))
    if atr is not None and hasattr(atr, "iloc"):
        atr = float(atr.iloc[-1]) if len(atr) else np.nan
    if atr is not None and not (pd.isna(atr) or atr <= 0):
        indicator_logger.info(
            "ATR provided by %s: %.6f",
            getattr(strategy_fn, "__name__", str(strategy_fn)),
            atr,
        )

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
                prev_score = score
                score *= bias
                score = max(0.0, min(score, 1.0))
                indicator_logger.info(
                    "Cycle bias %.2f adjusted score %.2f -> %.2f",
                    bias,
                    prev_score,
                    score,
                )
            except Exception:
                pass

    return score, direction, atr


async def evaluate_async(
    strategy_fns: List[Callable[[pd.DataFrame], Tuple]],
    df: pd.DataFrame,
    config: Optional[dict] = None,
    max_parallel: Optional[int] = 4,
) -> List[Tuple[float, str, Optional[float]]]:
    """Asynchronously evaluate strategy callables with limited concurrency."""

    if config is not None:
        cfg_mp = config.get("max_parallel")
        if cfg_mp is not None:
            try:
                max_parallel = int(cfg_mp)
            except Exception:
                pass

    if max_parallel is not None:
        if not isinstance(max_parallel, int) or max_parallel < 1:
            raise ValueError("max_parallel must be a positive integer or None")
        sem = asyncio.Semaphore(max_parallel)
    else:
        sem = asyncio.Semaphore(len(strategy_fns))

    async def run(fn: Callable[[pd.DataFrame], Tuple]):
        async with sem:
            async def call():
                return await asyncio.to_thread(evaluate, fn, df, config)

            return await asyncio.wait_for(call(), timeout=5)

    tasks: List[asyncio.Task] = []
    placeholders: List[int] = []
    executed_fns: List[Callable[[pd.DataFrame], Tuple]] = []
    results: List[Tuple[float, str, Optional[float]] | None] = []

    for fn in strategy_fns:
        min_bars = getattr(fn, "min_bars", 0)
        if min_bars and len(df) < int(min_bars):
            results.append((0.0, "none", None))
            continue
        placeholders.append(len(results))
        tasks.append(asyncio.create_task(run(fn)))
        executed_fns.append(fn)
        results.append(None)

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, fn, res in zip(placeholders, executed_fns, raw_results):
        if isinstance(res, Exception):
            if isinstance(res, asyncio.TimeoutError):
                logger.warning(
                    "Strategy %s TIMEOUT", getattr(fn, "__name__", str(fn))
                )
            else:
                logger.warning(
                    "Strategy %s failed: %s",
                    getattr(fn, "__name__", str(fn)),
                    res,
                )
            results[idx] = (0.0, "none", None)
        else:
            results[idx] = res

    # at this point all placeholders replaced
    return [r if r is not None else (0.0, "none", None) for r in results]


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
            logger.warning(
                "Strategy %s failed: %s",
                getattr(strat, "__name__", getattr(getattr(strat, "func", None), "__name__", str(strat))),
                exc,
            )
            continue

        metric = score + sharpe + drawdown
        if metric > best_score:
            best_score = metric
            best_res = {
                "score": score,
                "direction": direction,
                "name": getattr(strat, "__name__", getattr(getattr(strat, "func", None), "__name__", "")),
            }

    return best_res
