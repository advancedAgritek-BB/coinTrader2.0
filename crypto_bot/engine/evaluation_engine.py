import asyncio
import time
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import logging
from crypto_bot.strategies.loader import load_strategies

logger = logging.getLogger(__name__)


class EvalGate:
    """Async gate to serialize evaluations with TTL watchdog."""

    def __init__(self, logger: Any, ttl_sec: int = 120) -> None:
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

    async def start_watchdog(self) -> asyncio.Task:
        async def _watch() -> None:
            while True:
                await asyncio.sleep(5)
                if self._lock.locked() and self._since > 0:
                    held = time.monotonic() - self._since
                    if held > self._ttl:
                        self._logger.warning(
                            f"Gate held >{self._ttl}s by {self._owner}; forcing release"
                        )
                        try:
                            self._lock.release()
                        except RuntimeError:
                            pass
                        self._owner = None
                        self._since = 0.0

        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(_watch())
        return self._watchdog_task

    async def stop_watchdog(self) -> None:
        if self._watchdog_task is None:
            return
        self._watchdog_task.cancel()
        with suppress(Exception):
            await self._watchdog_task
        self._watchdog_task = None


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
        self._workers: list[asyncio.Task] = []
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self.gate: EvalGate | None = None
        self.strategies: list[Any] = []

    async def start(self) -> None:
        # discover and instantiate strategies based on mode
        mode = getattr(getattr(self.cfg, "trading", None), "mode", "auto")
        # auto-discover strategy modules without requiring an explicit list
        self.strategies, errors = load_strategies(mode, return_errors=True)
        if not self.strategies:
            err_msg = "\n".join(f"{name}: {err}" for name, err in errors.items())
            msg = "Aborting evaluator start: 0 strategies loaded." + (
                f"\n{err_msg}" if err_msg else ""
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if self.data is None:
            self.data = SimpleNamespace(ready=lambda symbol, tf: True)

        ttl = 120
        if self.cfg is not None and getattr(self.cfg, "evaluation", None) is not None:
            ttl = getattr(self.cfg.evaluation, "gate_ttl_sec", ttl)
        self.gate = EvalGate(logger, ttl_sec=ttl)
        await self.gate.start_watchdog()

        self._stop.clear()
        self._workers = []
        self._tasks = []
        workers = getattr(
            getattr(self.cfg, "evaluation", None), "workers", self.concurrency
        )
        for idx in range(workers):
            task = asyncio.create_task(self._worker(), name=f"eval-worker-{idx}")
            self._workers.append(task)
        logger.info(f"Evaluation workers online: {len(self._workers)}")

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
                f"STRAT {res.get('name')} on {symbol}: signal={signal} score={res.get('score')} reason={res.get('reason', '')}"
            )
        logger.debug(f"[EVAL OK] {symbol}")

    async def _worker(self) -> None:
        if self.gate is None or self.data is None:
            logger.error("Evaluation worker started before engine initialization")
            return
        logger.info("Evaluation worker online")
        while not self._stop.is_set():
            try:
                symbol, ctx = await self.queue.get()
                try:
                    needs_5m = self._symbol_requires_5m(ctx)
                    if not self.data.ready(symbol, "1m"):
                        logger.debug(f"EVAL SKIP {symbol}: 1m warmup not met")
                        continue
                    if needs_5m and not self.data.ready(symbol, "5m"):
                        logger.debug(f"EVAL SKIP {symbol}: 5m warmup not met")
                        continue

                    async with self.gate.hold(f"{symbol}"):
                        logger.info(f"EVAL START {symbol}")
                        await self._evaluate_symbol(symbol, ctx)
                except Exception:
                    logger.exception(f"Evaluator crashed on {symbol}")
                finally:
                    self.queue.task_done()
            except asyncio.CancelledError:  # pragma: no cover - shutdown
                return
            except Exception as e:
                logger.error(f"Worker error: {e}; restarting...")
                await asyncio.sleep(1)
        await asyncio.sleep(0)

    async def enqueue(self, symbol: str, ctx: dict) -> None:
        await self.queue.put((symbol, ctx))

    async def drain(self) -> None:
        await self.queue.join()

    async def stop(self) -> None:
        if not self._workers and not self._tasks:
            return
        self._stop.set()
        for t in self._workers + self._tasks:
            t.cancel()
        await asyncio.gather(*self._workers, *self._tasks, return_exceptions=True)
        self._workers.clear()
        self._tasks.clear()
        if self.gate:
            await self.gate.stop_watchdog()
        logger.info("Evaluation workers stopped")


_STREAM_EVAL: StreamEvaluationEngine | None = None


def set_stream_evaluator(evaluator: StreamEvaluationEngine) -> None:
    global _STREAM_EVAL
    _STREAM_EVAL = evaluator


def get_stream_evaluator() -> StreamEvaluationEngine:
    if _STREAM_EVAL is None:
        raise RuntimeError("EvaluationEngine not initialized")
    return _STREAM_EVAL

