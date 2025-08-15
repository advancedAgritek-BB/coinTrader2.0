import asyncio
import logging
from typing import Any, Dict, Iterable

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 8.0


async def run_strategies_for_symbol(symbol: str, ctx) -> Any:
    """Invoke the evaluation function stored on ``ctx`` for ``symbol``."""
    eval_fn = getattr(ctx, "eval_fn", None)
    if eval_fn is None:
        return None
    return await eval_fn(symbol)


async def evaluate_batch(symbols: Iterable[str], ctx) -> Dict[str, Any]:
    """Evaluate symbols sequentially with per-symbol timeout and logging."""
    results: Dict[str, Any] = {}

    for symbol in symbols:
        try:
            res = await asyncio.wait_for(
                run_strategies_for_symbol(symbol, ctx), timeout=DEFAULT_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"[EVAL TIMEOUT] {symbol}")
            continue
        except Exception as e:  # pragma: no cover - log and continue
            logger.exception(f"[EVAL ERROR] {symbol}: {e}")
            continue
        else:
            if isinstance(res, dict) and not res.get("skip"):
                results[symbol] = res

    return results
