from __future__ import annotations
import asyncio
import logging
from typing import Callable, Awaitable, Dict, Any

logger = logging.getLogger(__name__)

class StreamEvaluator:
    """
    Push symbols into an asyncio.Queue as soon as their OHLCV warmup is ready.
    One or more worker tasks consume and run strategy evaluation immediately.
    """
    def __init__(self, eval_fn: Callable[[str, Dict[str, Any]], Awaitable[None]], concurrency: int = 8):
        self.queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self.eval_fn = eval_fn
        self.concurrency = concurrency
        self._workers: list[asyncio.Task] = []
        self._closed = asyncio.Event()

    async def start(self):
        for i in range(self.concurrency):
            self._workers.append(asyncio.create_task(self._worker(i)))

    async def _worker(self, idx: int):
        while not self._closed.is_set():
            try:
                symbol, ctx = await self.queue.get()
            except asyncio.CancelledError:
                return
            try:
                await self.eval_fn(symbol, ctx)
                logger.debug(f"[EVAL OK] {symbol}")
            except asyncio.TimeoutError:
                logger.warning(f"[EVAL TIMEOUT] {symbol}")
            except Exception as e:
                logger.exception(f"[EVAL ERROR] {symbol}: {e}")
            finally:
                self.queue.task_done()

    async def enqueue(self, symbol: str, ctx: dict):
        await self.queue.put((symbol, ctx))

    async def drain(self):
        await self.queue.join()

    async def stop(self):
        self._closed.set()
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

STREAM_EVALUATOR: StreamEvaluator | None = None

def set_stream_evaluator(evaluator: StreamEvaluator) -> None:
    global STREAM_EVALUATOR
    STREAM_EVALUATOR = evaluator

def get_stream_evaluator() -> StreamEvaluator:
    if STREAM_EVALUATOR is None:
        raise RuntimeError("StreamEvaluator not initialized")
    return STREAM_EVALUATOR
