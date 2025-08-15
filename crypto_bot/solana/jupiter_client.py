import os
import aiohttp
import asyncio
from typing import Any, Dict, Optional
from loguru import logger

JUPITER_BASE = os.getenv("JUPITER_BASE_URL", "https://api.jup.ag")


class JupiterClient:
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
        assert self._session is not None, "JupiterClient used outside of async context manager"
        return self._session

    async def quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int = 150) -> Optional[Dict[str, Any]]:
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(slippage_bps),
            "onlyDirectRoutes": "false",
        }
        try:
            async with self.session.get(f"{JUPITER_BASE}/v6/quote", params=params) as r:
                if r.status != 200:
                    logger.debug(f"Jupiter quote HTTP {r.status}")
                    return None
                data = await r.json()
                routes = data.get("data") or []
                return routes[0] if routes else None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Jupiter.quote failed: {e!r}")
            return None

    async def swap_instructions(self, route: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            async with self.session.post(f"{JUPITER_BASE}/v6/swap-instructions", json={"route": route}) as r:
                if r.status != 200:
                    return None
                return await r.json()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Jupiter.swap_instructions failed: {e!r}")
            return None
