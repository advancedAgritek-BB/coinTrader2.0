from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StrategyState:
    """Track per-strategy diagnostics information."""

    missing_prereqs: Dict[str, List[str]] = field(default_factory=dict)
    last_signal: Dict[str, Optional[str]] = field(default_factory=dict)


class Gate:
    """Simple gate used to serialize evaluation work.

    The gate records who is currently holding it and for how long so
    diagnostics can report if evaluation is stuck.
    """

    def __init__(self) -> None:
        self.owner: Optional[str] = None
        self.held_since: Optional[float] = None

    def is_busy(self) -> bool:
        return self.owner is not None

    def status(self) -> dict:
        held = 0.0
        if self.is_busy() and self.held_since is not None:
            held = max(0.0, time.time() - self.held_since)
        return {
            "busy": self.is_busy(),
            "owner": self.owner,
            "held_seconds": held,
        }

    def hold(self, owner: str) -> None:
        self.owner = owner
        self.held_since = time.time()

    def release(self) -> None:
        self.owner = None
        self.held_since = None


class EvaluationEngine:
    """Book-keeping around strategy evaluation.

    Only a very small subset of the real system is implemented here to
    support the ``diagnose eval`` CLI command.  The engine keeps track of
    warmup candles, evaluation workers and strategy state.  The diagnostics
    surface why a symbol is not currently ready for trading.
    """

    def __init__(
        self,
        *,
        warmup_candles: Optional[Dict[str, int]] = None,
    ) -> None:
        # timeframe -> symbol -> list of candle dicts containing a "ts" key
        self.candles: Dict[str, Dict[str, List[Dict[str, int]]]] = {}
        self.warmup_candles = warmup_candles or {}
        self.gate = Gate()
        self.queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.strategies: Dict[str, StrategyState] = {}

    # ------------------------------------------------------------------
    # Warmup helpers
    # ------------------------------------------------------------------
    def _warmup_status(self, symbol: str) -> dict:
        result: Dict[str, dict] = {}
        missing: List[str] = []
        for tf in ("1m", "5m"):
            candles = self.candles.get(tf, {}).get(symbol, [])
            last_ts = candles[-1]["ts"] if candles else None
            required = int(self.warmup_candles.get(tf, 0) or 0)
            ready = candles is not None and len(candles) >= required and required > 0
            if not ready:
                missing.append(tf)
            result[tf] = {"ready": ready, "last_ts": last_ts}
        result["missing"] = missing
        return result

    def _queue_status(self) -> dict:
        alive = sum(1 for w in self.workers if not getattr(w, "done", lambda: True)())
        return {"length": self.queue.qsize(), "workers_alive": alive}

    def _strategy_status(self, symbol: str) -> List[dict]:
        info: List[dict] = []
        for name, state in self.strategies.items():
            missing = state.missing_prereqs.get(symbol, [])
            last = state.last_signal.get(symbol)
            info.append({"name": name, "missing": missing, "last_signal": last})
        return info

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def diagnose(self, symbol: str) -> str:
        """Return a human readable readiness report for ``symbol``."""

        lines = [f"Symbol: {symbol}"]

        warm = self._warmup_status(symbol)
        for tf in ("1m", "5m"):
            tf_info = warm.get(tf, {})
            status = "ready" if tf_info.get("ready") else "missing"
            ts = tf_info.get("last_ts")
            if ts is not None:
                lines.append(f"  {tf}: {status} last_ts={ts}")
            else:
                lines.append(f"  {tf}: {status}")
        if warm.get("missing"):
            lines.append(f"  missing_tf: {', '.join(warm['missing'])}")

        gate = self.gate.status()
        gate_line = (
            f"Gate: {'busy' if gate['busy'] else 'free'}"
        )
        if gate.get("owner"):
            gate_line += f" owner={gate['owner']}"
        if gate.get("held_seconds"):
            gate_line += f" held={gate['held_seconds']:.1f}s"
        lines.append(gate_line)

        q = self._queue_status()
        lines.append(f"Queue: {q['length']} pending, workers_alive={q['workers_alive']}")

        for s in self._strategy_status(symbol):
            missing = ",".join(s["missing"]) if s["missing"] else "ok"
            last = s["last_signal"] if s["last_signal"] is not None else "none"
            lines.append(f"{s['name']}: prereqs={missing} last={last}")

        return "\n".join(lines)


# Global engine instance for CLI use ----------------------------------------------------
ENGINE: Optional[EvaluationEngine] = None


def set_engine(engine: EvaluationEngine) -> None:
    global ENGINE
    ENGINE = engine


def get_engine() -> EvaluationEngine:
    if ENGINE is None:
        raise RuntimeError("Evaluation engine not configured")
    return ENGINE
