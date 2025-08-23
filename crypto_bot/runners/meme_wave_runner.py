from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Mapping

from crypto_bot.core.pipeline import scoring_loop
from crypto_bot.core.execution import execution_loop
from crypto_bot.core.queues import trade_queue
from crypto_bot.strategies import get_ohlcv_df
from crypto_bot.strategies.loader import load_strategies


async def run(config: Mapping[str, Any]) -> None:
    """Run the meme-wave trading pipeline.

    The function loads all enabled strategies for the requested mode and then
    scores each strategy for all ``(symbol, timeframe)`` pairs defined in the
    configuration.  Trade candidates are enqueued via :func:`scoring_loop` and
    consumed concurrently by :func:`execution_loop`.
    """

    mode = config.get("mode", "cex")
    strat_cfg = config.get("strategies", {})
    enabled = [name for name, s in strat_cfg.items() if s.get("enabled")]
    strategies = load_strategies(mode, enabled)

    symbols = config.get("symbols", [])
    timeframes = config.get("timeframes", [])

    exec_task = asyncio.create_task(execution_loop(config))
    try:
        score_tasks: list[asyncio.Task] = []
        for strat in strategies:
            for sym in symbols:
                for tf in timeframes:
                    ohlcv_df = await get_ohlcv_df(sym, tf)
                    score_tasks.append(
                        asyncio.create_task(
                            scoring_loop(config, strat, sym, tf, ohlcv_df)
                        )
                    )
        if score_tasks:
            await asyncio.gather(*score_tasks)
        await trade_queue.join()
    finally:
        exec_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await exec_task


__all__ = ["run"]
