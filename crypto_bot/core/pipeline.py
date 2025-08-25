import logging
from typing import Any, Mapping, Dict, Tuple

from .queues import trade_queue

log = logging.getLogger(__name__)


# Track last score/side per strategy-symbol-timeframe to suppress repetitive
# "no signal" log entries.
_last_signals: Dict[Tuple[str, str, str], Tuple[float, str]] = {}


async def scoring_loop(config, strategy, symbol: str, timeframe: str, ohlcv) -> None:
    """Score a strategy and enqueue trade candidates."""
    side, score, meta = strategy.generate_signal(symbol, timeframe, ohlcv)
    key = (strategy.name, symbol, timeframe)
    prev_score, prev_side = _last_signals.get(key, (None, None))
    if not (score == 0.0 and side == "none" and prev_score == 0.0 and prev_side == "none"):
        log.info(
            "Signal for %s | %s | %s: %.6f, %s",
            strategy.name,
            symbol,
            timeframe,
            score,
            side,
        )
    _last_signals[key] = (score, side)
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

