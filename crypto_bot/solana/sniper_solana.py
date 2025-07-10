from __future__ import annotations

from typing import Mapping, Tuple

import pandas as pd

from .risk import RiskTracker
from .safety import is_safe
from .score import score_event
from .watcher import NewPoolEvent


def generate_signal(df: pd.DataFrame, config: dict | None = None) -> Tuple[float, str]:
    """Return a neutral signal placeholder for Solana sniping."""
    return 0.0, "none"


def score_new_pool(
    event: NewPoolEvent,
    config: Mapping[str, object],
    risk_tracker: RiskTracker,
) -> Tuple[float, str]:
    """Return a score and direction for a new pool event.

    Parameters
    ----------
    event:
        Pool creation event to evaluate.
    config:
        Configuration mapping with ``scoring``, ``safety`` and ``risk``
        subsections. Optionally includes ``twitter_score``.
    risk_tracker:
        Tracker for enforcing risk limits.
    """

    if not is_safe(event, config.get("safety", {})):
        return 0.0, "none"

    if not risk_tracker.allow_snipe(event.token_mint, config.get("risk", {})):
        return 0.0, "none"

    scoring_cfg = config.get("scoring", {})
    score = score_event(event, scoring_cfg)

    sentiment = float(config.get("twitter_score", 0))
    weight = float(scoring_cfg.get("twitter_weight", 1.0))
    score += sentiment * weight

    return score, "long"
