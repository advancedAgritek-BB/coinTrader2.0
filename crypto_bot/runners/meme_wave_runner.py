from __future__ import annotations

import asyncio
from typing import Mapping

from crypto_bot.core.pipeline import scoring_loop
from crypto_bot.core.execution import execution_loop
from crypto_bot.strategy import load_strategies


async def run(
    config: Mapping[str, object],
    strategy,
    symbol: str,
    timeframe: str,
    ohlcv,
) -> None:
    """Run the meme-wave trading pipeline for a single strategy.

    Parameters
    ----------
    config:
        Runtime configuration passed to the scoring and execution loops.
    strategy:
        Strategy object exposing ``generate_signal``.
    symbol/timeframe/ohlcv:
        Inputs forwarded to :func:`scoring_loop`.
    """

    # Load strategies or perform other warm-up tasks before starting loops
    load_strategies(config.get("mode", "cex"))

    score_task = asyncio.create_task(
        scoring_loop(config, strategy, symbol, timeframe, ohlcv)
    )
    exec_task = asyncio.create_task(execution_loop(config))

    await asyncio.gather(score_task, exec_task)


__all__ = ["run"]
