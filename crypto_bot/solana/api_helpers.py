"""Async Solana API helpers."""

from __future__ import annotations

import aiohttp
from contextlib import asynccontextmanager

# Base endpoints from the blueprint
# Helius WebSocket: wss://mainnet.helius-rpc.com
# Jito Block Engine REST: https://mainnet.block-engine.jito.wtf/api/v1
# The blueprint recommends staying below 100 requests/sec for Helius and 60 requests/sec for Jito.
# API keys are provided via environment variables HELIUS_KEY and JITO_KEY.


@asynccontextmanager
async def helius_ws(api_key: str):
    """Yield a websocket connection to Helius RPC and close the session."""

    url = f"wss://mainnet.helius-rpc.com/?api-key={api_key}"
    session = aiohttp.ClientSession()
    ws = await session.ws_connect(url)
    try:
        yield ws
    finally:
        await ws.close()
        await session.close()


async def fetch_jito_bundle(bundle_id: str, api_key: str, session: aiohttp.ClientSession | None = None):
    """Fetch a bundle status from Jito Block Engine."""

    close = False
    if session is None:
        session = aiohttp.ClientSession()
        close = True
    async with session.get(
        f"https://mainnet.block-engine.jito.wtf/api/v1/bundles/{bundle_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    if close:
        await session.close()
    return data
