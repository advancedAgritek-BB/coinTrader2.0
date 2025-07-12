"""Async Solana API helpers."""

from __future__ import annotations

import aiohttp
from contextlib import asynccontextmanager

# Base endpoints from the blueprint
# Helius WebSocket: wss://rpc.helius.xyz
# Jito Block Engine REST: https://mainnet.block-engine.jito.wtf/api/v1
# The blueprint recommends staying below 100 requests/sec for Helius and 60 requests/sec for Jito.
# API keys are provided via environment variables HELIUS_KEY and JITO_KEY.


@asynccontextmanager
async def helius_ws(api_key: str):
    """Yield a websocket connection to Helius RPC and close the session."""

    url = f"wss://rpc.helius.xyz/?api-key={api_key}"
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

import os
import requests
from crypto_bot.fund_manager import TOKEN_MINTS

JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"


def fetch_solana_price(symbol: str, url: str = JUPITER_QUOTE_URL) -> float:
    """Return token price on Solana DEXes using the Jupiter aggregator."""
    mock = os.getenv("MOCK_SOLANA_PRICE")
    if mock is not None:
        try:
            return float(mock)
        except ValueError:
            return 0.0
    try:
        base, quote = symbol.split("/")
    except ValueError:
        return 0.0
    base_mint = TOKEN_MINTS.get(base)
    quote_mint = TOKEN_MINTS.get(quote)
    if base_mint is None or quote_mint is None:
        return 0.0
    try:
        resp = requests.get(
            url,
            params={"inputMint": base_mint, "outputMint": quote_mint, "amount": 1},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("data"):
            route = data["data"][0]
            return float(route.get("outAmount", 0)) / float(route.get("inAmount", 1))
    except Exception:
        pass
    return 0.0

