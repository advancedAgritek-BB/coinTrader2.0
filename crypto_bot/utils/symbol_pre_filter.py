import os
import json
import asyncio
from typing import Iterable, List

import aiohttp

from .logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/symbol_filter.log")

API_URL = "https://api.kraken.com/0/public"


async def _fetch_ticker_async(pairs: Iterable[str]) -> dict:
    """Return ticker data for ``pairs`` in batches of 20 symbols using aiohttp."""

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
    """Return volume in USD, percent change and bid/ask spread from ``ticker``."""
    last = float(ticker["c"][0])
    open_price = float(ticker.get("o", last))
    volume = float(ticker["v"][1])
    vwap = float(ticker["p"][1])
    ask = float(ticker["a"][0])
    bid = float(ticker["b"][0])

    volume_usd = volume * vwap
    change_pct = ((last - open_price) / open_price) * 100 if open_price else 0.0
    spread = abs(ask - bid) / last * 100 if last else 0.0
    return volume_usd, change_pct, spread


async def filter_symbols(exchange, symbols: Iterable[str]) -> List[str]:
    """Return subset of ``symbols`` passing liquidity and volatility checks."""
    pairs = [s.replace("/", "") for s in symbols]
    data = (await _fetch_ticker_async(pairs)).get("result", {})
    id_map = {}
    if hasattr(exchange, "markets_by_id"):
        if not exchange.markets_by_id and hasattr(exchange, "load_markets"):
            try:
                exchange.load_markets()
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("load_markets failed: %s", exc)
        id_map = {}
        for k, v in exchange.markets_by_id.items():
            if isinstance(v, dict):
                id_map[k] = v.get("symbol", k)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                id_map[k] = v[0].get("symbol", k)
            else:
                id_map[k] = k
    allowed: List[str] = []
    for pair_id, ticker in data.items():
        symbol = id_map.get(pair_id)
        if not symbol:
            # fallback match by stripped pair
            for sym in symbols:
                if pair_id.upper() == sym.replace("/", "").upper():
                    symbol = sym
                    break
        if not symbol:
            continue
        vol_usd, change_pct, spread = _parse_metrics(ticker)
        logger.info(
            "Ticker %s volume %.2f USD change %.2f%% spread %.2f%%",
            symbol,
            vol_usd,
            change_pct,
            spread,
        )
        if vol_usd > 50000 and abs(change_pct) > 1:
            allowed.append(symbol)
    return allowed
