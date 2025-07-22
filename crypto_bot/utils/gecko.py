from __future__ import annotations

import asyncio
import logging

import aiohttp

logger = logging.getLogger(__name__)


async def gecko_request(
    url: str,
    params: dict | None = None,
    retries: int = 3,
) -> list | dict | None:
    """Return JSON from GeckoTerminal ``url`` with retry logic."""

    for attempt in range(max(1, retries)):
        # wait before request as instructed
        await asyncio.sleep(1 + attempt)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(10 * (2 ** attempt))
                        continue
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as exc:
            logger.error("Gecko request failed for %s: %s", url, exc)
    return None
