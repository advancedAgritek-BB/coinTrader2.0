import logging
from typing import Any, Mapping

from .queues import trade_queue

log = logging.getLogger(__name__)


async def scoring_loop(config, strategy, symbol: str, timeframe: str, ohlcv) -> None:
    """Score a strategy and enqueue trade candidates."""
    side, score, meta = strategy.signal(symbol, timeframe, ohlcv)
    log.info(
        "Signal for %s | %s | %s: %.6f, %s",
        strategy.name,
        symbol,
        timeframe,
        score,
        side,
    )
    if isinstance(config, Mapping):
        thresholds = config.get("thresholds", {})  # type: ignore[assignment]
    else:
        thresholds = getattr(config, "thresholds", {})
    if side != "none" and score >= thresholds.get(strategy.name, {}).get(timeframe, 0.02):
        cand = {
            "symbol": symbol,
            "side": side,
            "score": score,
            "strategy": strategy.name,
            "timeframe": timeframe,
            "meta": meta,
        }
        trade_queue.put_nowait(cand)
        log.info(
            "ENQUEUED %s %s %s score=%.4f",
            strategy.name,
            symbol,
            timeframe,
            score,
        )

