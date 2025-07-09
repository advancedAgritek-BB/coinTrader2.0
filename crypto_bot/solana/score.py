"""Scoring utilities for Solana pool events."""

from __future__ import annotations

from typing import Mapping

from .watcher import NewPoolEvent


def score_event(event: NewPoolEvent, cfg: Mapping[str, float]) -> float:
    """Return a numeric score for ``event`` based on heuristic weights."""

    liq_weight = float(cfg.get("weight_liquidity", 1.0))
    tx_weight = float(cfg.get("weight_tx", 1.0))
    social_weight = float(cfg.get("weight_social", 1.0))
    rug_weight = float(cfg.get("weight_rug", 1.0))

    liquidity_score = event.liquidity * liq_weight
    tx_score = event.tx_count * tx_weight
    social_score = float(cfg.get("social_score", 0)) * social_weight
    rug_penalty = float(cfg.get("rug_risk", 0)) * rug_weight

    return liquidity_score + tx_score + social_score - rug_penalty
