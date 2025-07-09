"""Async Solana API helpers."""

from __future__ import annotations

import aiohttp

# Base endpoints from the blueprint
# Helius WebSocket: wss://rpc.helius.xyz
# Jito Block Engine REST: https://mainnet.block-engine.jito.wtf/api/v1
# The blueprint recommends staying below 100 requests/sec for Helius and 60 requests/sec for Jito.
# API keys are provided via environment variables HELIUS_KEY and JITO_KEY.


async def connect_helius_ws(api_key: str):
    """Return a websocket connection to Helius RPC."""

    url = f"wss://rpc.helius.xyz/?api-key={api_key}"
    session = aiohttp.ClientSession()
    ws = await session.ws_connect(url)
    return ws


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
