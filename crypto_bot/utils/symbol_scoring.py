"""Helpers for scoring market symbols."""

from __future__ import annotations

from typing import Mapping
import time

DEFAULT_WEIGHTS = {
    "volume": 0.4,
    "change": 0.2,
    "spread": 0.2,
    "age": 0.1,
    "latency": 0.1,
}


def get_symbol_age(exchange, symbol: str) -> float:
    """Return age of ``symbol`` in days using ``exchange.markets``."""

    try:
        markets = getattr(exchange, "markets", None)
        if not markets and hasattr(exchange, "load_markets"):
            markets = exchange.load_markets()
        market = markets.get(symbol) if markets else None
        if not market:
            return 0.0

        ts = (
            market.get("created")
            or market.get("timestamp")
            or market.get("info", {}).get("listed")
            or market.get("info", {}).get("launch")
        )
        if ts is None:
            return 0.0

        now = (
            exchange.milliseconds() if hasattr(exchange, "milliseconds") else int(time.time() * 1000)
        )
        return max(0.0, (now - int(ts)) / 86400000)
    except Exception:  # pragma: no cover - best effort
        return 0.0


def get_latency(exchange, symbol: str) -> float:
    """Return recent API latency for ``symbol`` in milliseconds."""

    if not hasattr(exchange, "fetch_ticker"):
        return 0.0

    start = time.perf_counter()
    try:
        exchange.fetch_ticker(symbol)
    except Exception:  # pragma: no cover - best effort
        pass
    end = time.perf_counter()
    return (end - start) * 1000.0


def score_symbol(
    exchange,
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
    age_norm = min(get_symbol_age(exchange, symbol) / max_age, 1.0)
    latency_norm = 1.0 - min(get_latency(exchange, symbol) / max_latency, 1.0)

    score = (
        volume_norm * weights.get("volume", 0)
        + change_norm * weights.get("change", 0)
        + spread_norm * weights.get("spread", 0)
        + age_norm * weights.get("age", 0)
        + latency_norm * weights.get("latency", 0)
    )

    return score / total

