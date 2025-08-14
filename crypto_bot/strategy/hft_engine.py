from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Literal

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.execution.cex_executor import execute_trade_async


# Placeholder async generator patched in tests or by runtime code.  It yields
# market snapshots consumed by :class:`HFTEngine`.  The default implementation
# is empty so importing this module does not attempt any network access.
async def stream_snapshots(*_a, **_k):  # pragma: no cover - replaced elsewhere
    if False:
        yield {}


@dataclass
class Signal:
    """Simple trade instruction produced by strategies."""

    symbol: str
    side: Literal["buy", "sell"]
    price: float
    size: float
    reason: str
    ttl_ms: int = 1000


class _DummyNotifier:
    """Minimal notifier used when running in dry-run mode."""

    def __init__(self, logger):
        self._logger = logger

    def notify(self, msg: str) -> None:
        self._logger.info(msg)
        return None


class HFTEngine:
    """Dispatch real-time snapshots to microstructure strategies."""

    def __init__(
        self,
        exchange: Any,
        fees_cfg: Optional[Dict],
        symbols: List[str],
        strategies: List[Callable[[Dict], Optional[Signal]]],
    ) -> None:
        self.exchange = exchange
        self.fees_cfg = fees_cfg or {}
        self.symbols = symbols
        self.strategies = strategies
        tele_cfg = self.fees_cfg.get("telemetry", {})
        self.batch_summary_secs = tele_cfg.get("batch_summary_secs", 60)
        self.dry_run = bool(self.fees_cfg.get("dry_run", True))

        self.logger = setup_logger(__name__, LOG_DIR / "hft_engine.log")
        self.notifier = _DummyNotifier(self.logger)

        # Active signals keyed by symbol.  Each entry is a mapping containing:
        # ``task`` - execution asyncio.Task, ``priority`` - strategy index,
        # ``ts`` - timestamp when scheduled, ``signal`` - the Signal object.
        self._active: Dict[str, Dict[str, Any]] = {}

        # Counters for periodic logging
        self._strategy_counts: Counter = Counter()
        self._suppress_counts: Counter = Counter()
        self._last_summary = time.monotonic()

    async def _execute(self, sig: Signal) -> None:
        cfg = {
            "post_only": getattr(sig, "post_only", False),
            "taker": getattr(sig, "taker", False),
        }
        await execute_trade_async(
            self.exchange,
            None,
            sig.symbol,
            sig.side,
            sig.size,
            notifier=self.notifier,
            dry_run=self.dry_run,
            config=cfg,
        )

    async def run(self) -> None:
        """Consume snapshot stream and dispatch to strategies."""

        async for snap in stream_snapshots(self.exchange, self.symbols):
            now = time.monotonic()

            # Cleanup completed or expired signals
            for sym, info in list(self._active.items()):
                task = info["task"]
                sig: Signal = info["signal"]
                if task.done():
                    del self._active[sym]
                    continue
                if now - info["ts"] > sig.ttl_ms / 1000:
                    task.cancel()
                    self._suppress_counts["ttl_expired"] += 1
                    del self._active[sym]

            for idx, strat in enumerate(self.strategies):
                try:
                    sig = strat(snap)
                except Exception:  # pragma: no cover - strategy errors
                    self.logger.exception(
                        "Strategy %s failed", getattr(strat, "__name__", idx)
                    )
                    continue
                if sig is None:
                    continue

                strat_name = getattr(strat, "__name__", f"strategy_{idx}")
                current = self._active.get(sig.symbol)
                if current and current["priority"] <= idx:
                    self._suppress_counts["lower_priority"] += 1
                    continue
                if current:
                    current["task"].cancel()
                    self._suppress_counts["replaced"] += 1
                    del self._active[sig.symbol]

                task = asyncio.create_task(self._execute(sig))
                self._active[sig.symbol] = {
                    "task": task,
                    "priority": idx,
                    "ts": now,
                    "signal": sig,
                }
                self._strategy_counts[strat_name] += 1

            if now - self._last_summary >= self.batch_summary_secs:
                if self._strategy_counts or self._suppress_counts:
                    self.logger.info(
                        "Batch summary: signals=%s suppressed=%s",
                        dict(self._strategy_counts),
                        dict(self._suppress_counts),
                    )
                self._strategy_counts.clear()
                self._suppress_counts.clear()
                self._last_summary = now
