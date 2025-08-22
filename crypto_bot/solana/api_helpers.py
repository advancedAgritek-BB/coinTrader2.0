"""Async Solana API helpers."""

from __future__ import annotations

import aiohttp
from contextlib import asynccontextmanager
import os

from crypto_bot.utils.http_client import get_session

# Base endpoints from the blueprint
# Helius WebSocket: wss://mainnet.helius-rpc.com
# Jito Block Engine REST: https://mainnet.block-engine.jito.wtf/api/v1
# The blueprint recommends staying below 100 requests/sec for Helius and 60 requests/sec for Jito.
# API keys are provided via environment variables HELIUS_KEY and JITO_KEY.


@asynccontextmanager
async def helius_ws(
    api_key: str | None = None,
    session: aiohttp.ClientSession | None = None,
):
    """Yield a websocket connection to Helius RPC and close the session.

    If ``api_key`` is not provided, the value of the ``HELIUS_KEY``
    environment variable will be used.

    Raises
    ------
    RuntimeError
        If no API key is available.
    """

    if api_key is None:
        api_key = os.getenv("HELIUS_KEY")
    if not api_key:
        raise RuntimeError("HELIUS API key is required; set api_key or HELIUS_KEY env var")

    url = f"wss://mainnet.helius-rpc.com/?api-key={api_key}"
    session = session or get_session()
    ws = await session.ws_connect(url)
    try:
        yield ws
    finally:
        await ws.close()


async def fetch_jito_bundle(
    bundle_id: str,
    api_key: str,
    session: aiohttp.ClientSession | None = None,
):
    """Fetch a bundle status from Jito Block Engine."""

    session = session or get_session()
    async with session.get(
        f"https://mainnet.block-engine.jito.wtf/api/v1/bundles/{bundle_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data
