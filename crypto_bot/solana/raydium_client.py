import aiohttp
import asyncio
from typing import Any, Dict, Optional
from loguru import logger

RAYDIUM_BASE = "https://api.raydium.io"


class RaydiumClient:
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
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
        assert self._session is not None, "RaydiumClient used outside of async context manager"
        return self._session

    async def pools_by_mint(self, mint: str) -> list[Dict[str, Any]]:
        try:
            async with self.session.get(f"{RAYDIUM_BASE}/v2/main/pairs", params={"mint1": mint}) as r:
                if r.status != 200:
                    return []
                data = await r.json()
                return data if isinstance(data, list) else []
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Raydium.pools_by_mint failed: {e!r}")
            return []

    async def best_pool_for_mint(self, mint: str, *, min_liquidity_usd: float = 5000.0) -> Optional[Dict[str, Any]]:
        pools = await self.pools_by_mint(mint)
        candidates = [p for p in pools if float(p.get("liquidityUsd", 0) or 0) >= min_liquidity_usd]
        if not candidates:
            return None
        candidates.sort(key=lambda p: float(p.get("volume24hUsd", 0) or 0), reverse=True)
        return candidates[0]
