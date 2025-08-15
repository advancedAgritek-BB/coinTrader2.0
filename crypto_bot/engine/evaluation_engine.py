import asyncio
import logging
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Awaitable, Callable


logger = logging.getLogger(__name__)


class EvalGate:
    """Async gate to serialize evaluations with TTL watchdog."""

    def __init__(self, logger: logging.Logger, ttl_sec: int = 120) -> None:
        self._lock = asyncio.Lock()
        self._owner: str | None = None
        self._since: float = 0.0
        self._logger = logger
        self._ttl = ttl_sec
        self._watchdog_task: asyncio.Task | None = None

    @asynccontextmanager
    async def hold(self, owner: str):
        await self._lock.acquire()
        self._owner = owner
        self._since = time.monotonic()
        try:
            yield
        finally:
            self._owner = None
            self._since = 0.0
            self._lock.release()

    async def start_watchdog(self) -> None:
        async def _watch() -> None:
            while True:
                await asyncio.sleep(5)
                if self._lock.locked() and self._since > 0:
                    held = time.monotonic() - self._since
                    if held > self._ttl:
                        self._logger.warning(
                            "Gate held >%ss by %s; forcing release",
                            self._ttl,
                            self._owner,
                        )
                        try:
                            self._lock.release()
                        except RuntimeError:
                            pass
                        self._owner = None
                        self._since = 0.0

        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(_watch())


class StreamEvaluationEngine:
    """Queue based evaluation engine with worker pool logging."""

    def __init__(
        self,
        eval_fn: Callable[[str, dict], Awaitable[Any]],
        concurrency: int = 8,
        cfg: Any | None = None,
        data: Any | None = None,
    ) -> None:
        self.queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self.eval_fn = eval_fn
        self.concurrency = concurrency
        self._workers: list[asyncio.Task] = []
        self._closed = asyncio.Event()
        self.data = data or SimpleNamespace(ready=lambda symbol, tf: True)
        ttl = 120
        if cfg is not None and getattr(cfg, "evaluation", None) is not None:
            ttl = getattr(cfg.evaluation, "gate_ttl_sec", ttl)
        self.gate = EvalGate(logger, ttl_sec=ttl)

    async def start(self) -> None:
        await self.gate.start_watchdog()
        for _ in range(self.concurrency):
            self._workers.append(asyncio.create_task(self._worker()))
        logger.info(
            "Evaluation workers online: %d; queue=%d",
            self.concurrency,
            self.queue.qsize(),
        )

    def _symbol_requires_5m(self, ctx: dict) -> bool:
        timeframes = []
        if isinstance(ctx, dict):
            timeframes = ctx.get("timeframes", [])
        return "5m" in timeframes

    async def _evaluate_symbol(self, symbol: str, ctx: dict) -> None:
        timeframes = []
        if isinstance(ctx, dict):
            timeframes = ctx.get("timeframes", [])
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

    async def _worker(self) -> None:
        logger.info("Evaluation worker online")
        while not self._closed.is_set():
            try:
                symbol, ctx = await self.queue.get()
            except asyncio.CancelledError:  # pragma: no cover - shutdown
                return
            try:
                needs_5m = self._symbol_requires_5m(ctx)
                if not self.data.ready(symbol, "1m"):
                    logger.debug("EVAL SKIP %s: 1m warmup not met", symbol)
                    continue
                if needs_5m and not self.data.ready(symbol, "5m"):
                    logger.debug("EVAL SKIP %s: 5m warmup not met", symbol)
                    continue

                async with self.gate.hold(f"{symbol}"):
                    logger.info("EVAL START %s", symbol)
                    await self._evaluate_symbol(symbol, ctx)
            except Exception:
                logger.exception("Evaluator crashed on %s", symbol)
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

