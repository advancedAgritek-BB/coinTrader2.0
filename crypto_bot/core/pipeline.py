"""Core scoring loop for strategy evaluation.

This module previously expected strategy objects to expose a ``signal``
coroutine returning ``(side, score, meta)``.  The majority of strategies in
this repository, however, only expose a synchronous ``generate_signal``
function which returns ``(score, side)``.  As a result the original interface
caused attribute errors and prevented strategies from feeding trade
recommendations into the pipeline.

The scoring loop now adapts to the ``generate_signal`` style interface used by
``mean_bot`` and ``momentum_bot``.  It also honours per‑strategy score
thresholds supplied via ``config['thresholds']``.
"""

from __future__ import annotations

import logging
from typing import Mapping

from .queues import trade_queue

log = logging.getLogger(__name__)


async def scoring_loop(config: Mapping[str, object], strategy, symbol: str, timeframe: str, ohlcv) -> None:
    """Score ``strategy`` on ``ohlcv`` data and enqueue trade candidates.

    Parameters
    ----------
    config:
        Mapping that may contain per‑strategy configurations and a
        ``"thresholds"`` mapping of minimum scores.
    strategy:
        Object exposing ``name`` and ``generate_signal`` attributes.
    symbol:
        Trading pair symbol.
    timeframe:
        Candle timeframe (e.g. ``"1h"``).
    ohlcv:
        DataFrame containing OHLCV candles for the symbol/timeframe.
    """

    strat_cfg = None
    if isinstance(config, Mapping):
        strat_cfg = config.get(getattr(strategy, "name", "")) or config
        thresholds = config.get("thresholds", {})
    else:  # pragma: no cover - defensive, Mapping expected
        strat_cfg = getattr(config, getattr(strategy, "name", ""), config)
        thresholds = getattr(config, "thresholds", {})

    score, side = strategy.generate_signal(
        ohlcv, symbol=symbol, timeframe=timeframe, config=strat_cfg
    )
    log.info(
        "Signal for %s | %s | %s: %.6f, %s",
        strategy.name,
        symbol,
        timeframe,
        score,
        side,
    )

    min_score = thresholds.get(getattr(strategy, "name", ""), {}).get(timeframe, 0.02)
    if side != "none" and score >= min_score:
        cand = {
            "symbol": symbol,
            "side": side,
            "score": score,
            "strategy": strategy.name,
            "timeframe": timeframe,
            "meta": {},
        }
        trade_queue.put_nowait(cand)
        log.info(
            "ENQUEUED %s %s %s score=%.4f",
            strategy.name,
            symbol,
            timeframe,
            score,
        )

