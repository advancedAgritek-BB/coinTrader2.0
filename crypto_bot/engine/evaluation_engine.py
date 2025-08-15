import asyncio
import logging
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from crypto_bot.strategy import load_strategies

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

    def is_busy(self) -> bool:
        """Return whether the gate is currently held."""
        return self._lock.locked()

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
        self.cfg = cfg
        self.data = data
        self._running = False
        self._workers: list[asyncio.Task] = []
        self._tasks: list[asyncio.Task] = []
        self._closed: asyncio.Event | None = None
        self.gate: EvalGate | None = None
        self.strategies: dict[str, Any] = {}
        self.strategy_import_errors: dict[str, str] = {}

    async def start(self) -> None:
        # read enabled list from config if available
        enabled = set(getattr(self.cfg, "strategies", {}).get("enabled", [])) or None
        self.strategies, self.strategy_import_errors = load_strategies(
            enabled=enabled
        )
        if not self.strategies:
            logger.error(
                "Aborting evaluator start: 0 strategies loaded. See above import errors."
            )
            return

        if self.data is None:
            self.data = SimpleNamespace(ready=lambda symbol, tf: True)

        ttl = 120
        if self.cfg is not None and getattr(self.cfg, "evaluation", None) is not None:
            ttl = getattr(self.cfg.evaluation, "gate_ttl_sec", ttl)
        self.gate = EvalGate(logger, ttl_sec=ttl)
        await self.gate.start_watchdog()

        self._running = True
        self._closed = asyncio.Event()
        self._workers = []
        self._tasks = []
        workers = getattr(
            getattr(self.cfg, "evaluation", None), "workers", self.concurrency
        )
        for idx in range(workers):
            task = asyncio.create_task(self._worker(), name=f"eval-worker-{idx}")
            self._workers.append(task)
        logger.info("Evaluation workers online: %d", len(self._workers))

    def _symbol_requires_5m(self, ctx: dict) -> bool:
        if isinstance(ctx, dict):
            return "5m" in ctx.get("timeframes", [])
        return False

    async def _evaluate_symbol(self, symbol: str, ctx: dict) -> None:
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
        if self.gate is None or self._closed is None or self.data is None:
            logger.error("Evaluation worker started before engine initialization")
            return
        while self._running:
            logger.info("Evaluation worker online")
            while self._running and not self._closed.is_set():
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
            await asyncio.sleep(0)

    async def enqueue(self, symbol: str, ctx: dict) -> None:
        await self.queue.put((symbol, ctx))

    async def drain(self) -> None:
        await self.queue.join()

    async def stop(self) -> None:
        self._running = False
        if self._closed:
            self._closed.set()
        for t in self._workers + self._tasks:
            t.cancel()
        for t in self._workers + self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._workers.clear()
        self._tasks.clear()
        logger.info("Evaluation workers stopped")


_STREAM_EVAL: StreamEvaluationEngine | None = None


def set_stream_evaluator(evaluator: StreamEvaluationEngine) -> None:
    global _STREAM_EVAL
    _STREAM_EVAL = evaluator


def get_stream_evaluator() -> StreamEvaluationEngine:
    if _STREAM_EVAL is None:
        raise RuntimeError("EvaluationEngine not initialized")
    return _STREAM_EVAL

