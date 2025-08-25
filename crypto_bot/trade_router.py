import logging
from typing import Iterable, List, Dict, Any

logger = logging.getLogger(__name__)


def select(
    signals: Iterable[Dict[str, Any]],
    *,
    min_score: float = 0.0,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Filter and rank strategy ``signals`` into trade candidates.

    Parameters
    ----------
    signals:
        Iterable of signal dictionaries produced by :mod:`strategy_manager`.
    min_score:
        Minimum score a signal must have to be considered.
    limit:
        Optional maximum number of candidates to return.
    """

    if not signals:
        return []

    candidates: List[Dict[str, Any]] = []
    for sig in signals:
        try:
            score = float(sig.get("score", 0.0))
            direction = sig.get("direction") or sig.get("signal")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Skipping malformed signal %s: %s", sig, exc)
            continue
        if direction in {"long", "short"} and score >= min_score:
            candidates.append(sig)

    candidates.sort(key=lambda s: s.get("score", 0.0), reverse=True)
    if limit is not None:
        candidates = candidates[:limit]
    return candidates
