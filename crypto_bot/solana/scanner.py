from __future__ import annotations

"""Utilities for scanning new Solana tokens."""

import asyncio
import logging
import os
from typing import Mapping, List

import aiohttp

logger = logging.getLogger(__name__)


async def get_solana_new_tokens(cfg: Mapping[str, object]) -> List[str]:
    """Return a list of new token mint addresses using ``cfg`` options."""

    url = str(cfg.get("url", ""))
    if not url:
        return []

    key = os.getenv("HELIUS_KEY", "")
    if "${HELIUS_KEY}" in url:
        url = url.replace("${HELIUS_KEY}", key)
    if "YOUR_KEY" in url:
        url = url.replace("YOUR_KEY", key)

    limit = int(cfg.get("limit", 0))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        logger.error("Solana scanner error: %s", exc)
        return []

    tokens = data.get("tokens") or data.get("mints") or data
    results: List[str] = []
    if isinstance(tokens, list):
        for item in tokens:
            if isinstance(item, str):
                results.append(item)
            elif isinstance(item, Mapping):
                mint = item.get("mint") or item.get("tokenMint") or item.get("token_mint")
                if mint:
                    results.append(str(mint))
    elif isinstance(tokens, Mapping):
        for mint in tokens.values():
            if isinstance(mint, str):
                results.append(mint)
    if limit:
        results = results[:limit]
    return results
