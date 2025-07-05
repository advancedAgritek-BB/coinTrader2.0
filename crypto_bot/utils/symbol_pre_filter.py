"""Utility for filtering trading pairs based on liquidity and volatility."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Iterable, List

import numpy as np

import aiohttp

from .logger import setup_logger
from .market_loader import fetch_ohlcv_async
from .symbol_scoring import score_symbol


logger = setup_logger(__name__, "crypto_bot/logs/symbol_filter.log")

API_URL = "https://api.kraken.com/0/public"
DEFAULT_MIN_VOLUME_USD = 50000


async def has_enough_history(
    exchange, symbol: str, days: int = 30, timeframe: str = "1d"
) -> bool:
    """Return ``True`` when ``symbol`` has at least ``days`` days of history."""
    data = await fetch_ohlcv_async(
        exchange, symbol, timeframe=timeframe, limit=days
    )
    if not data:
        return False
    first_ts = data[0][0] / 1000
    last_ts = data[-1][0] / 1000
    return (last_ts - first_ts) / 86400 + 1 >= days



async def _fetch_ticker_async(pairs: Iterable[str]) -> dict:
    """Return ticker data for ``pairs`` in batches of 20 using aiohttp."""

    mock = os.getenv("MOCK_KRAKEN_TICKER")
    if mock:
        return json.loads(mock)

    pairs_list = list(pairs)
    combined: dict = {"result": {}, "error": []}
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(0, len(pairs_list), 20):
            chunk = pairs_list[i : i + 20]
            url = f"{API_URL}/Ticker?pair={','.join(chunk)}"
            tasks.append(session.get(url, timeout=10))

        responses = await asyncio.gather(*tasks)
        for resp in responses:
            resp.raise_for_status()
            data = await resp.json()
            combined["error"] += data.get("error", [])
            combined["result"].update(data.get("result", {}))

    return combined


def _parse_metrics(ticker: dict) -> tuple[float, float, float]:
    """Return volume USD, percent change and spread percentage."""

    last = float(ticker["c"][0])
    open_price = float(ticker.get("o", last))
    volume = float(ticker["v"][1])
    vwap = float(ticker["p"][1])
    ask = float(ticker["a"][0])
    bid = float(ticker["b"][0])

    volume_usd = volume * vwap
    change_pct = ((last - open_price) / open_price) * 100 if open_price else 0.0
    spread_pct = abs(ask - bid) / last * 100 if last else 0.0
    return volume_usd, change_pct, spread_pct


def _timeframe_seconds(exchange, timeframe: str) -> int:
    if hasattr(exchange, "parse_timeframe"):
        try:
            return exchange.parse_timeframe(timeframe)
        except Exception:
            pass
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    if unit == "w":
        return value * 604800
    if unit == "M":
        return value * 2592000
    raise ValueError(f"Unknown timeframe {timeframe}")


async def _has_enough_history(
    exchange, symbol: str, min_days: int, timeframe: str = "1h"
) -> bool:
    """Return ``True`` if ``symbol`` has at least ``min_days`` of OHLCV."""

    seconds = _timeframe_seconds(exchange, timeframe)
    candles_needed = int((min_days * 86400) / seconds) + 1

    try:
        data = await fetch_ohlcv_async(
            exchange, symbol, timeframe=timeframe, limit=candles_needed
        )
    except Exception as exc:  # pragma: no cover - network
        logger.warning("fetch_ohlcv failed for %s: %s", symbol, exc)
        return False

    if not data or len(data) < 2:
        return False

    first_ts = data[0][0]
    last_ts = data[-1][0]
    return last_ts - first_ts >= min_days * 86400 - seconds


async def filter_symbols(
    exchange, symbols: Iterable[str], config: dict | None = None
) -> List[str]:
    """Return subset of ``symbols`` passing liquidity checks sorted by score."""

    cfg = config or {}
    min_volume = cfg.get("symbol_filter", {}).get(
        "min_volume_usd", DEFAULT_MIN_VOLUME_USD
    )
    min_age = cfg.get("min_symbol_age_days", 0)
    min_score = cfg.get("min_symbol_score", 0.4)

    """Return subset of ``symbols`` passing liquidity and volatility checks.

    Parameters
    ----------
    exchange: object
        Exchange instance providing market metadata.
    symbols: Iterable[str]
        Pairs to evaluate.
    config: dict | None
        Optional configuration dictionary containing ``symbol_filter`` with
        ``min_volume_usd`` setting.
    """
    min_volume = DEFAULT_MIN_VOLUME_USD
    min_age = 0
    pct = 80
    if config:
        sf = config.get("symbol_filter", {})
        min_volume = sf.get("min_volume_usd", DEFAULT_MIN_VOLUME_USD)
        pct = sf.get("change_pct_percentile", 80)
        min_age = config.get("min_symbol_age_days", 0)
    pairs = [s.replace("/", "") for s in symbols]
    data = (await _fetch_ticker_async(pairs)).get("result", {})

    id_map: dict[str, str] = {}
    if hasattr(exchange, "markets_by_id"):
        if not exchange.markets_by_id and hasattr(exchange, "load_markets"):
            try:
                exchange.load_markets()
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("load_markets failed: %s", exc)
        for k, v in exchange.markets_by_id.items():
            if isinstance(v, dict):
                id_map[k] = v.get("symbol", k)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                id_map[k] = v[0].get("symbol", k)
            else:
                id_map[k] = v if isinstance(v, str) else k

    scored: List[tuple[str, float]] = []
                id_map[k] = k
    metrics: List[tuple[str, float, float, float]] = []
    for pair_id, ticker in data.items():
        symbol = id_map.get(pair_id)
        if not symbol:
            for sym in symbols:
                if pair_id.upper() == sym.replace("/", "").upper():
                    symbol = sym
                    break
        if not symbol:
            continue

        vol_usd, change_pct, spread_pct = _parse_metrics(ticker)
        logger.info(
            "Ticker %s volume %.2f USD change %.2f%% spread %.2f%%",
            symbol,
            vol_usd,
            change_pct,
            spread_pct,
        )

        if vol_usd < min_volume or abs(change_pct) <= 1:
            continue

        if min_age > 0:
            enough = await has_enough_history(exchange, symbol, min_age)
            if not enough:
                logger.info("Skipping %s due to insufficient history", symbol)
                continue

        score = score_symbol(symbol, vol_usd, change_pct, spread_pct, cfg)
        if score < min_score:
            logger.info("Skipping %s score %.3f below %.3f", symbol, score, min_score)
            continue

        scored.append((symbol, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in scored]

        metrics.append((symbol, vol_usd, change_pct, spread))

    if not metrics:
        return []

    threshold = np.percentile([abs(c[2]) for c in metrics], pct)

    allowed: List[tuple[str, float]] = []
    for symbol, vol_usd, change_pct, _ in metrics:
        if vol_usd > min_volume and abs(change_pct) >= threshold:
            if min_age > 0:
                enough = await _has_enough_history(exchange, symbol, min_age)
                if not enough:
                    logger.info("Skipping %s due to insufficient history", symbol)
                    continue
            allowed.append((symbol, vol_usd))

    allowed.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in allowed]
