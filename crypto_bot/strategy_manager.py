import asyncio
import logging
from typing import Iterable, List, Dict, Any

from .strategies.loader import load_strategies
from .strategies import score as _score

logger = logging.getLogger(__name__)


async def evaluate_all(
    symbols: Iterable[str],
    timeframes: Iterable[str],
    mode: str = "cex",
) -> List[Dict[str, Any]]:
    """Evaluate enabled strategies across ``symbols`` and ``timeframes``.

    Strategies are loaded via :func:`load_strategies` and each is scored using
    :func:`crypto_bot.strategies.score`.  Any strategy errors are logged and the
    strategy is skipped.  The returned list contains one entry per
    ``symbol/timeframe`` combination with ``score`` and ``direction`` fields.
    """

    strategies = load_strategies(mode)
    signals: List[Dict[str, Any]] = []

    for strat in strategies:
        name = getattr(strat, "__name__", str(strat))
        try:
            results = await _score(strat, symbols=symbols, timeframes=timeframes)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Strategy %s evaluation failed: %s", name, exc)
            continue

        for (symbol, timeframe), data in results.items():
            score_val = 0.0
            direction = "none"
            extra: Dict[str, Any] = {}
            if isinstance(data, dict):
                score_val = float(data.get("score", 0.0))
                direction = str(data.get("signal") or data.get("direction") or "none")
                extra = data.get("meta") or data.get("extra") or {}
            elif isinstance(data, tuple):
                if len(data) > 0:
                    score_val = float(data[0])
                if len(data) > 1:
                    direction = str(data[1])
                if len(data) > 2 and isinstance(data[2], dict):
                    extra = data[2]
            signals.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "score": score_val,
                    "direction": direction,
                    "extra": extra,
                }
            )

    return signals
