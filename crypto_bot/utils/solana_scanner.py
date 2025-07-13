from __future__ import annotations

"""Fetch newly launched Solana tokens from public APIs."""

import asyncio
import logging
from typing import List
import aiohttp

logger = logging.getLogger(__name__)

# Global min volume filter updated by ``get_solana_new_tokens``
_MIN_VOLUME_USD = 0.0

RAYDIUM_URL = "https://api.raydium.io/pairs"
PUMP_FUN_URL = "https://client-api.prod.pump.fun/v1/launches"


async def _fetch_json(url: str) -> list | dict | None:
    """Return parsed JSON from ``url`` using ``aiohttp``."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        logger.error("Solana scanner request failed: %s", exc)
        return None


def _extract_tokens(data: list | dict) -> List[str]:
    """Return token mints from ``data`` respecting ``_MIN_VOLUME_USD``."""
    items = data.get("data") if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        return []

    results: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mint = (
            item.get("tokenMint")
            or item.get("token_mint")
            or item.get("mint")
            or item.get("address")
        )
        if not mint:
            continue
        vol = (
            item.get("volumeUsd")
            or item.get("volume_usd")
            or item.get("liquidityUsd")
            or item.get("liquidity_usd")
            or 0.0
        )
        try:
            volume = float(vol)
        except Exception:
            volume = 0.0
        if volume >= _MIN_VOLUME_USD:
            results.append(str(mint))
    return results


async def fetch_new_raydium_pools(api_key: str, limit: int) -> List[str]:
    """Return new Raydium pool token mints."""
    url = f"{RAYDIUM_URL}?apiKey={api_key}&limit={limit}"
    data = await _fetch_json(url)
    if not data:
        return []
    tokens = _extract_tokens(data)
    return tokens[:limit]


async def fetch_pump_fun_launches(api_key: str, limit: int) -> List[str]:
    """Return recent Pump.fun launches."""
    url = f"{PUMP_FUN_URL}?api-key={api_key}&limit={limit}"
    data = await _fetch_json(url)
    if not data:
        return []
    tokens = _extract_tokens(data)
    return tokens[:limit]


async def get_solana_new_tokens(config: dict) -> List[str]:
    """Return deduplicated Solana token symbols from multiple sources."""

    global _MIN_VOLUME_USD

    limit = int(config.get("max_tokens_per_scan", 0)) or 20
    _MIN_VOLUME_USD = float(config.get("min_volume_usd", 0.0))
    raydium_key = str(config.get("raydium_api_key", ""))
    pump_key = str(config.get("pump_fun_api_key", ""))

    tasks = []
    if raydium_key:
        tasks.append(fetch_new_raydium_pools(raydium_key, limit))
    if pump_key:
        tasks.append(fetch_pump_fun_launches(pump_key, limit))

    if not tasks:
        return []

    results = await asyncio.gather(*tasks)
    mints: List[str] = []
    seen: set[str] = set()
    for res in results:
        for mint in res:
            if mint not in seen:
                seen.add(mint)
                mints.append(f"{mint}/USDC")
            if len(mints) >= limit:
                break
        if len(mints) >= limit:
            break

    return mints
