import asyncio
import logging
from typing import Awaitable, Callable, Any

logger = logging.getLogger(__name__)


class StreamEvaluationEngine:
    """Queue based evaluation engine with worker pool logging."""

    def __init__(self, eval_fn: Callable[[str, dict], Awaitable[Any]], concurrency: int = 8):
        self.queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self.eval_fn = eval_fn
        self.concurrency = concurrency
        self._workers: list[asyncio.Task] = []
        self._closed = asyncio.Event()

    async def start(self) -> None:
        for _ in range(self.concurrency):
            self._workers.append(asyncio.create_task(self._worker()))
        logger.info(
            "Evaluation workers online: %d; queue=%d",
            self.concurrency,
            self.queue.qsize(),
        )

    async def _worker(self) -> None:
        while not self._closed.is_set():
            try:
                symbol, ctx = await self.queue.get()
            except asyncio.CancelledError:  # pragma: no cover - shutdown
                return
            timeframes = []
            if isinstance(ctx, dict):
                timeframes = ctx.get("timeframes", [])
            logger.info("EVAL START %s on %s", symbol, timeframes)
            try:
                res = await asyncio.wait_for(self.eval_fn(symbol, ctx), timeout=8)
                if isinstance(res, dict):
                    direction = res.get("direction", "none")
                    signal = (
                        "BUY"
                        if direction == "long"
                        else "SELL" if direction == "short" else "NONE"
                    )
                    logger.info(
                        "STRAT %s on %s: signal=%s score=%s reason=%s",
                        res.get("name"),
                        symbol,
                        signal,
                        res.get("score"),
                        res.get("reason", ""),
                    )
                logger.debug("[EVAL OK] %s", symbol)
            except asyncio.TimeoutError:
                logger.warning("[EVAL TIMEOUT] %s", symbol)
            except Exception as exc:  # pragma: no cover - best effort
                logger.exception("[EVAL ERROR] %s: %s", symbol, exc)
            finally:
                self.queue.task_done()

    async def enqueue(self, symbol: str, ctx: dict) -> None:
        await self.queue.put((symbol, ctx))

    async def drain(self) -> None:
        await self.queue.join()

    async def stop(self) -> None:
        self._closed.set()
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)


_STREAM_EVAL: StreamEvaluationEngine | None = None


def set_stream_evaluator(evaluator: StreamEvaluationEngine) -> None:
    global _STREAM_EVAL
    _STREAM_EVAL = evaluator


def get_stream_evaluator() -> StreamEvaluationEngine:
    if _STREAM_EVAL is None:
        raise RuntimeError("EvaluationEngine not initialized")
    return _STREAM_EVAL
import threading
from typing import Any, Awaitable, Callable

from crypto_bot.utils.eval_guard import eval_gate

logger = logging.getLogger(__name__)


def _has_ohlcv(ctx: Any, symbol: str, timeframe: str, warmup_met: bool = True) -> bool:
    """Return True if OHLCV data for *symbol*/*timeframe* exists and meets warmup."""
    cache = getattr(ctx, "df_cache", {}) or {}
    tf_cache = cache.get(timeframe, {}) or {}
    df = tf_cache.get(symbol)
    if df is None or getattr(df, "empty", True):
        return False
    if warmup_met:
        warm_map = getattr(ctx, "config", {}).get("warmup_candles", {}) or {}
        required = int(warm_map.get(timeframe, 0) or 0)
        if required and len(df) < required:
            return False
    return True


class GatedEvaluationEngine:
    """Evaluate strategies with gate protection and warmup checks."""

    def __init__(self, ttl: float = 120.0):
        self.ttl = ttl

    async def _watchdog(self, symbol: str, strat: str) -> None:
        try:
            await asyncio.sleep(self.ttl)
        except asyncio.CancelledError:  # pragma: no cover - watchdog cancelled
            return
        if eval_gate.is_busy():
            logger.warning(
                "Gate held >%ds by %s/%s; force-releasing",
                self.ttl,
                symbol,
                strat,
            )
            eval_gate._busy = False  # force release

    async def evaluate(
        self,
        symbol: str,
        strategy: Callable[[str, Any], Awaitable[Any]],
        ctx: Any,
    ) -> Any:
        """Evaluate *strategy* for *symbol* using *ctx* with gate management."""
        if not _has_ohlcv(ctx, symbol, "1m", warmup_met=True):
            logger.debug(
                "EVAL SKIP %s: prerequisites not met (missing 5m or warmup)", symbol
            )
            return None
        needs_5m = False
        try:
            lookback = getattr(strategy, "required_lookback")()
            needs_5m = "5m" in (lookback or {})
        except Exception:  # pragma: no cover - strategy missing hook
            needs_5m = False
        if needs_5m and not _has_ohlcv(ctx, symbol, "5m", warmup_met=True):
            logger.debug(
                "EVAL SKIP %s: prerequisites not met (missing 5m or warmup)", symbol
            )
            return None

        strat_name = getattr(strategy, "__name__", str(strategy))
        owner = threading.get_ident()
        logger.debug("Gate acquire by %s/%s (owner=%s)", symbol, strat_name, owner)
        watchdog = asyncio.create_task(self._watchdog(symbol, strat_name))
        try:
            with eval_gate.hold(f"{symbol}/{strat_name}"):
                return await strategy(symbol, ctx)
        finally:
            watchdog.cancel()
            logger.debug(
                "Gate release by %s/%s (owner=%s)", symbol, strat_name, owner
            )
