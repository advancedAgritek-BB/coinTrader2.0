import os
import logging
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
