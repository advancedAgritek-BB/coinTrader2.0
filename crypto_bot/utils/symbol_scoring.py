"""Helpers for scoring market symbols."""

from __future__ import annotations

from typing import Mapping, Tuple, Dict
import asyncio
import inspect
import time
import numpy as np

try:  # pragma: no cover - optional dependency
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover - numba missing
    HAS_NUMBA = False

    def njit(*_a, **_k):  # type: ignore
        def wrap(fn):
            return fn

        return wrap
import pandas as pd

# Cache of computed symbol ages
_age_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}

# Cache of latency measurements keyed by exchange hostname
_latency_cache: Dict[str, Tuple[float, float]] = {}

# refresh intervals
_AGE_REFRESH = 24 * 3600  # 24 hours in seconds
_LATENCY_REFRESH = 10 * 60  # 10 minutes in seconds

DEFAULT_WEIGHTS = {
    "volume": 0.4,
    "change": 0.2,
    "spread": 0.2,
    "age": 0.1,
    "latency": 0.1,
    "liquidity": 0.0,
}


def get_symbol_age(exchange, symbol: str) -> float:
    """Return age of ``symbol`` in days using ``exchange.markets`` with caching."""

    key = (getattr(exchange, "id", exchange.__class__.__name__), symbol)
    now_s = time.time()
    cached = _age_cache.get(key)
    if cached and now_s - cached[1] < _AGE_REFRESH:
        return cached[0]

    try:
        markets = getattr(exchange, "markets", None)
        if not markets and hasattr(exchange, "load_markets"):
            markets = exchange.load_markets()
        market = markets.get(symbol) if markets else None
        if not market:
            age = 0.0
        else:
            ts = market.get("created")
            if not ts:
                ts = market.get("timestamp")
            if not ts:
                ts = market.get("info", {}).get("listed")
            if not ts:
                ts = market.get("info", {}).get("launch")
            if not ts:
                age = 0.0
            else:
                now_ms = (
                    exchange.milliseconds() if hasattr(exchange, "milliseconds") else int(time.time() * 1000)
                )
                age = max(0.0, (now_ms - int(ts)) / 86400000)
    except Exception:  # pragma: no cover - best effort
        age = 0.0

    _age_cache[key] = (age, now_s)
    return age


async def get_latency(exchange, symbol: str) -> float:
    """Return recent API latency for ``symbol`` in milliseconds using a cache."""

    if not hasattr(exchange, "fetch_ticker"):
        return 0.0

    host = getattr(exchange, "hostname", getattr(exchange, "id", ""))
    now_s = time.time()
    cached = _latency_cache.get(host)
    if cached and now_s - cached[1] < _LATENCY_REFRESH:
        return cached[0]

    start = time.perf_counter()
    try:
        fetch = exchange.fetch_ticker
        if inspect.iscoroutinefunction(fetch):
            await fetch(symbol)
        else:
            await asyncio.to_thread(fetch, symbol)
    except Exception:  # pragma: no cover - best effort
        pass
    end = time.perf_counter()
    latency = (end - start) * 1000.0

    _latency_cache[host] = (latency, now_s)
    return latency


def _score_vectorised_py(
    volume_usd: np.ndarray,
    change_pct: np.ndarray,
    spread_pct: np.ndarray,
    age_days: np.ndarray,
    latency_ms: np.ndarray,
    liquidity: np.ndarray,
    weights: np.ndarray,
    max_vals: np.ndarray,
    total: float,
) -> np.ndarray:
    volume_norm = np.minimum(volume_usd / max_vals[0], 1.0)
    change_norm = np.minimum(np.abs(change_pct) / max_vals[1], 1.0)
    spread_norm = 1.0 - np.minimum(spread_pct / max_vals[2], 1.0)
    age_norm = np.minimum(age_days / max_vals[3], 1.0)
    latency_norm = 1.0 - np.minimum(latency_ms / max_vals[4], 1.0)
    liq_norm = np.minimum(liquidity, 1.0)
    score = (
        volume_norm * weights[0]
        + change_norm * weights[1]
        + spread_norm * weights[2]
        + age_norm * weights[3]
        + latency_norm * weights[4]
        + liq_norm * weights[5]
    )
    return score / total


@njit(cache=True)
def _score_vectorised_numba(
    volume_usd,
    change_pct,
    spread_pct,
    age_days,
    latency_ms,
    liquidity,
    weights,
    max_vals,
    total,
):  # pragma: no cover - compiled
    n = len(volume_usd)
    out = np.empty(n)
    for i in range(n):
        v_norm = volume_usd[i] / max_vals[0]
        if v_norm > 1.0:
            v_norm = 1.0
        c_norm = abs(change_pct[i]) / max_vals[1]
        if c_norm > 1.0:
            c_norm = 1.0
        s_norm = 1.0 - spread_pct[i] / max_vals[2]
        if s_norm < 0.0:
            s_norm = 0.0
        a_norm = age_days[i] / max_vals[3]
        if a_norm > 1.0:
            a_norm = 1.0
        l_norm = 1.0 - latency_ms[i] / max_vals[4]
        if l_norm < 0.0:
            l_norm = 0.0
        liq_norm = liquidity[i]
        if liq_norm > 1.0:
            liq_norm = 1.0
        val = (
            v_norm * weights[0]
            + c_norm * weights[1]
            + s_norm * weights[2]
            + a_norm * weights[3]
            + l_norm * weights[4]
            + liq_norm * weights[5]
        )
        out[i] = val / total
    return out


def score_vectorised(
    volume_usd: np.ndarray,
    change_pct: np.ndarray,
    spread_pct: np.ndarray,
    age_days: np.ndarray,
    latency_ms: np.ndarray,
    liquidity: np.ndarray,
    config: Mapping[str, object],
) -> np.ndarray:
    """Vectorised symbol scoring with optional numba acceleration."""

    weights_dict = dict(DEFAULT_WEIGHTS)
    weights_dict.update(config.get("symbol_score_weights", {}))
    weight_arr = np.array(
        [
            weights_dict.get("volume", 0.0),
            weights_dict.get("change", 0.0),
            weights_dict.get("spread", 0.0),
            weights_dict.get("age", 0.0),
            weights_dict.get("latency", 0.0),
            weights_dict.get("liquidity", 0.0),
        ],
        dtype=np.float64,
    )
    total = float(weight_arr.sum())
    if total <= 0:
        raise ValueError("symbol_score_weights must sum to a positive value")

    max_vals = np.array(
        [
            float(config.get("max_vol", 1_000_000)),
            float(config.get("max_change_pct", 10)),
            float(config.get("max_spread_pct", 2)),
            float(config.get("max_age_days", 180)),
            float(config.get("max_latency_ms", 1000)),
            1.0,
        ],
        dtype=np.float64,
    )

    use_numba = bool(config.get("use_numba_scoring"))
    if use_numba and HAS_NUMBA:
        return _score_vectorised_numba(
            volume_usd,
            change_pct,
            spread_pct,
            age_days,
            latency_ms,
            liquidity,
            weight_arr,
            max_vals,
            total,
        )
    return _score_vectorised_py(
        volume_usd,
        change_pct,
        spread_pct,
        age_days,
        latency_ms,
        liquidity,
        weight_arr,
        max_vals,
        total,
    )


async def score_symbol(
    exchange,
    symbol: str,
    volume_usd: float,
    change_pct: float,
    spread_pct: float,
    liquidity: float,
    config: Mapping[str, object],
) -> float:
    """Return a normalized score for ``symbol``."""

    weights = dict(DEFAULT_WEIGHTS)
    weights.update(config.get("symbol_score_weights", {}))
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("symbol_score_weights must sum to a positive value")

    max_vol = float(config.get("max_vol", 1_000_000))
    max_change = float(config.get("max_change_pct", 10))
    max_spread = float(config.get("max_spread_pct", 2))
    max_age = float(config.get("max_age_days", 180))
    max_latency = float(config.get("max_latency_ms", 1000))

    volume_norm = min(volume_usd / max_vol, 1.0)
    change_norm = min(abs(change_pct) / max_change, 1.0)
    spread_norm = 1.0 - min(spread_pct / max_spread, 1.0)
    age_norm = min(get_symbol_age(exchange, symbol) / max_age, 1.0)
    latency_norm = 1.0 - min(await get_latency(exchange, symbol) / max_latency, 1.0)
    liq_norm = min(liquidity, 1.0)

    score = (
        volume_norm * weights.get("volume", 0)
        + change_norm * weights.get("change", 0)
        + spread_norm * weights.get("spread", 0)
        + age_norm * weights.get("age", 0)
        + latency_norm * weights.get("latency", 0)
        + liq_norm * weights.get("liquidity", 0)
    )

    return score / total


def score_vectorised(df: pd.DataFrame, config: Mapping[str, object]) -> pd.Series:
    """Return normalized scores for each row in ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``vol`` (volume USD), ``chg`` (percent change) and
        ``spr`` (spread percent) columns.
    config : Mapping[str, object]
        Configuration with ``symbol_score_weights`` and limit values.
    """

    weights = dict(DEFAULT_WEIGHTS)
    weights.update(config.get("symbol_score_weights", {}))
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("symbol_score_weights must sum to a positive value")

    max_vol = float(config.get("max_vol", 1_000_000))
    max_change = float(config.get("max_change_pct", 10))
    max_spread = float(config.get("max_spread_pct", 2))

    volume_norm = np.minimum(df["vol"] / max_vol, 1.0)
    change_norm = np.minimum(df["chg"].abs() / max_change, 1.0)
    spread_norm = 1.0 - np.minimum(df["spr"] / max_spread, 1.0)
    liq_norm = np.minimum(df.get("liq", 1.0), 1.0)

    score = (
        volume_norm * weights.get("volume", 0)
        + change_norm * weights.get("change", 0)
        + spread_norm * weights.get("spread", 0)
        + liq_norm * weights.get("liquidity", 0)
    )

    # Age and latency are ignored by the vectorised implementation. Set the
    # weights to zero or fall back to :func:`score_symbol` for full scoring.
    return score / total

