from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Iterable


@dataclass
class BotContext:
    """Shared state for bot phases."""

    positions: dict
    df_cache: dict
    regime_cache: dict
    config: dict
    exchange: object | None = None
    secondary_exchange: object | None = None
    ws_client: object | None = None
    risk_manager: object | None = None
    notifier: object | None = None
    wallet: object | None = None
    paper_wallet: object | None = None  # legacy alias
    position_guard: object | None = None
    balance: float = 0.0
    current_batch: list[str] = field(default_factory=list)
    analysis_results: list | None = field(default_factory=list)
    timing: dict | None = field(default_factory=dict)
    volatility_factor: float = 1.0
    mempool_monitor: object | None = None
    mempool_cfg: dict | None = None
    active_universe: list[str] = field(default_factory=list)
    resolved_mode: str = "auto"


class PhaseRunner:
    """Run a sequence of async phases and record timing."""

    def __init__(self, phases: Iterable[Callable[[BotContext], Awaitable[None]]]):
        self.phases = list(phases)

    async def run(self, ctx: BotContext) -> Dict[str, float]:
        timings: Dict[str, float] = {}
        for phase in self.phases:
            start = time.perf_counter()
            await phase(ctx)
            timings[phase.__name__] = time.perf_counter() - start
        return timings
