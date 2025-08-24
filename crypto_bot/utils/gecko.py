from __future__ import annotations

import asyncio
import logging

import aiohttp
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .http_client import get_session

logger = logging.getLogger(__name__)


async def gecko_request(
    url: str,
    params: dict | None = None,
    retries: int = 3,
) -> list | dict:
    """Return JSON from GeckoTerminal ``url`` with retry logic."""

    session = get_session()
    async for attempt in AsyncRetrying(
        wait=wait_exponential(multiplier=1),
        stop=stop_after_attempt(max(1, retries)),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 429:
                    raise aiohttp.ClientResponseError(
                        resp.request_info,
                        resp.history,
                        status=resp.status,
                        message="Too Many Requests",
                        headers=resp.headers,
                    )
                resp.raise_for_status()
                return await resp.json()
