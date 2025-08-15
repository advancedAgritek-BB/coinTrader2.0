import asyncio
import logging
from typing import Awaitable, Callable, Any

logger = logging.getLogger(__name__)


class EvaluationEngine:
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


_STREAM_EVAL: EvaluationEngine | None = None


def set_stream_evaluator(evaluator: EvaluationEngine) -> None:
    global _STREAM_EVAL
    _STREAM_EVAL = evaluator


def get_stream_evaluator() -> EvaluationEngine:
    if _STREAM_EVAL is None:
        raise RuntimeError("EvaluationEngine not initialized")
    return _STREAM_EVAL
