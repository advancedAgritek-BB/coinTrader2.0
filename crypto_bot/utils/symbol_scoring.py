"""Helpers for scoring market symbols."""

from __future__ import annotations

from typing import Mapping

DEFAULT_WEIGHTS = {
    "volume": 0.4,
    "change": 0.2,
    "spread": 0.2,
    "age": 0.1,
    "latency": 0.1,
}


def get_symbol_age(symbol: str) -> float:
    """Return age of ``symbol`` in days (stub)."""

    # Real implementation would query the exchange listing date
    return 0.0


def get_latency(symbol: str) -> float:
    """Return recent API latency for ``symbol`` in milliseconds (stub)."""

    return 0.0


def score_symbol(
    symbol: str,
    volume_usd: float,
    change_pct: float,
    spread_pct: float,
    config: Mapping[str, object],
) -> float:
    """Return a normalized score for ``symbol``."""

    weights = dict(DEFAULT_WEIGHTS)
    weights.update(config.get("symbol_score_weights", {}))
    total = sum(weights.values()) or 1.0

    max_vol = float(config.get("max_vol", 1_000_000))
    max_change = float(config.get("max_change_pct", 10))
    max_spread = float(config.get("max_spread_pct", 2))
    max_age = float(config.get("max_age_days", 180))
    max_latency = float(config.get("max_latency_ms", 1000))

    volume_norm = min(volume_usd / max_vol, 1.0)
    change_norm = min(abs(change_pct) / max_change, 1.0)
    spread_norm = 1.0 - min(spread_pct / max_spread, 1.0)
    age_norm = min(get_symbol_age(symbol) / max_age, 1.0)
    latency_norm = 1.0 - min(get_latency(symbol) / max_latency, 1.0)

    score = (
        volume_norm * weights.get("volume", 0)
        + change_norm * weights.get("change", 0)
        + spread_norm * weights.get("spread", 0)
        + age_norm * weights.get("age", 0)
        + latency_norm * weights.get("latency", 0)
    )

    return score / total

