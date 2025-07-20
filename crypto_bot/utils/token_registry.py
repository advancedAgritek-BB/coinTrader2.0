import json
import logging
from pathlib import Path

import aiohttp

TOKEN_REGISTRY_URL = (
    "https://raw.githubusercontent.com/solana-labs/token-list/main/src/tokens/solana.tokenlist.json"
)

CACHE_FILE = Path(__file__).resolve().parents[2] / "cache" / "token_mints.json"

logger = logging.getLogger(__name__)


async def load_token_mints() -> dict[str, str]:
    """Return a mapping of token symbol to mint address."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(TOKEN_REGISTRY_URL, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
        tokens = data.get("tokens") or []
        results: dict[str, str] = {}
        for item in tokens:
            sym = item.get("symbol")
            mint = item.get("address") or item.get("mint") or item.get("tokenMint")
            if sym and mint:
                results[str(sym)] = str(mint)
        if results:
            try:
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(CACHE_FILE, "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as exc:  # pragma: no cover - optional cache
                logger.error("Failed to write %s: %s", CACHE_FILE, exc)
        return results
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Token registry fetch failed: %s", exc)
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE) as f:
                    cached = json.load(f)
                if isinstance(cached, dict):
                    return {str(k): str(v) for k, v in cached.items()}
            except Exception as err:  # pragma: no cover - best effort
                logger.error("Failed to read cache: %s", err)
        return {}


TOKEN_MINTS: dict[str, str] = {}
"""Token registry utilities for Solana assets used by Jupiter."""

from __future__ import annotations

import logging
import os
from typing import Dict

import aiohttp

logger = logging.getLogger(__name__)

# Mapping of symbol to Solana token mint used by Jupiter
TOKEN_MINTS: Dict[str, str] = {
    "BTC": "So11111111111111111111111111111111111111112",
    "ETH": "2NdXGW7dpwye9Heq7qL3gFYYUUDewfxCUUDq36zzfrqD",
    "USDC": "EPjFWdd5AufqSSqeM2q6ksjLpaEweidnGj9n92gtQgNf",
    "SOL": "So11111111111111111111111111111111111111112",
}

_DEFAULT_URL = (
    "https://raw.githubusercontent.com/solana-labs/token-list/main/src/tokens/solana.tokenlist.json"
)

_LOADED = False


async def load_token_mints(url: str | None = None) -> Dict[str, str]:
    """Return mapping of token symbols to mint addresses.

    The list is fetched from ``url`` or ``TOKEN_MINTS_URL`` environment variable.
    Subsequent calls return an empty dict to avoid re-downloading.
    """
    global _LOADED
    if _LOADED:
        return {}
    url = url or os.getenv("TOKEN_MINTS_URL", _DEFAULT_URL)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Failed to fetch token registry: %s", exc)
        return {}

    tokens = data.get("tokens") or data.get("data", {}).get("tokens") or []
    result: Dict[str, str] = {}
    for item in tokens:
        symbol = item.get("symbol") or item.get("ticker")
        mint = item.get("address") or item.get("mint")
        if isinstance(symbol, str) and isinstance(mint, str):
            result[symbol.upper()] = mint
    _LOADED = True
    return result

def set_token_mints(mapping: dict[str, str]) -> None:
    """Replace ``TOKEN_MINTS`` with ``mapping`` after normalizing keys."""
    TOKEN_MINTS.clear()
    TOKEN_MINTS.update({k.upper(): v for k, v in mapping.items()})

