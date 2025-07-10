from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Iterable


@dataclass
class BotContext:
    """Shared state for bot phases."""

    positions: dict
    df_cache: dict
    regime_cache: dict
    config: dict


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
