from __future__ import annotations

import asyncio
from typing import Mapping

from crypto_bot.core.pipeline import scoring_loop
from crypto_bot.core.execution import execution_loop
from crypto_bot.strategy import load_strategies


async def run(config: Mapping[str, object]) -> None:
    """Run the meme-wave trading pipeline."""
    # Load strategies or perform other warm-up tasks before starting loops
    load_strategies(config.get("mode", "cex"))

    score_task = asyncio.create_task(scoring_loop(config))
    exec_task = asyncio.create_task(execution_loop(config))

    await asyncio.gather(score_task, exec_task)


__all__ = ["run"]
