import os
import aiohttp
import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger

HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")
HELIUS_BASE = "https://api.helius.xyz"


class HeliusClient:
    def __init__(self, api_key: Optional[str] = None, session: Optional[aiohttp.ClientSession] = None):
        self.api_key = api_key or HELIUS_API_KEY
        if not self.api_key:
            logger.warning("HELIUS_API_KEY not set; on-chain metadata will be unavailable.")
        self._ext_session = session
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        if self._ext_session:
            self._session = self._ext_session
        else:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if not self._ext_session and self._session:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        assert self._session is not None, "HeliusClient used outside of async context manager"
        return self._session

    async def get_token_metadata(self, mints: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Batch fetch token metadata. Returns {mint: metadata}.
        """
        if not self.api_key:
            return {}
        url = f"{HELIUS_BASE}/v0/tokens/metadata?api-key={self.api_key}"
        payload = {"mintAccounts": mints, "includeOffChain": True}
        try:
            async with self.session.post(url, json=payload) as r:
                if r.status != 200:
                    txt = await r.text()
                    logger.warning(f"Helius metadata HTTP {r.status}: {txt[:200]}")
                    return {}
                data = await r.json()
                out: Dict[str, Dict[str, Any]] = {}
                for meta in data or []:
                    mint = meta.get("mint")
                    if mint:
                        out[mint] = meta
                return out
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Helius.get_token_metadata failed: {e!r}")
            return {}


def helius_available() -> bool:
    """Lightweight check that the Helius API key is set and service responds."""
    if not HELIUS_API_KEY:
        return False
    url = f"{HELIUS_BASE}/v0/addresses?api-key={HELIUS_API_KEY}"

    async def _check() -> bool:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(url) as resp:
                    return resp.status in (200, 400, 404)
        except Exception:
            return False

    try:
        return asyncio.run(_check())
    except Exception:
        return False
