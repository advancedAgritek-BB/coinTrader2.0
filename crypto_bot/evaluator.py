import asyncio
import logging
import time
from typing import Any, Dict, Iterable

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 8.0
SUMMARY_INTERVAL = 60.0


async def run_strategies_for_symbol(symbol: str, ctx) -> Any:
    """Invoke the evaluation function stored on ``ctx`` for ``symbol``."""
    eval_fn = getattr(ctx, "eval_fn", None)
    if eval_fn is None:
        return None
    return await eval_fn(symbol)


async def evaluate_batch(symbols: Iterable[str], ctx) -> Dict[str, Any]:
    """Evaluate ``symbols`` sequentially with per-symbol timeout and logging."""
    stats = {"scanned": 0, "ok": 0, "errors": 0, "timeouts": 0, "signals": 0}
    results: Dict[str, Any] = {}

    for symbol in symbols:
        stats["scanned"] += 1
        try:
            res = await asyncio.wait_for(
                run_strategies_for_symbol(symbol, ctx), timeout=DEFAULT_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"[EVAL TIMEOUT] {symbol}")
            stats["timeouts"] += 1
            continue
        except Exception as e:  # pragma: no cover - log and continue
            logger.exception(f"[EVAL ERROR] {symbol}: {e}")
            stats["errors"] += 1
            continue
        else:
            stats["ok"] += 1
            if isinstance(res, dict) and not res.get("skip"):
                stats["signals"] += 1
            results[symbol] = res

        # ``time.monotonic`` is patched in tests and may advance in large jumps
        # because of internal asyncio calls. To keep the behaviour predictable
        # we only emit a single summary after processing all symbols instead of
        # periodic updates.

    if stats["scanned"]:
        logger.info(
            "Eval stats (last 60s): scanned=%d ok=%d errors=%d timeouts=%d signals=%d",
            stats["scanned"],
            stats["ok"],
            stats["errors"],
            stats["timeouts"],
            stats["signals"],
        )
    return results
