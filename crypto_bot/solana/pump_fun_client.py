from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import httpx

# Pump.fun API tends to flap; use backoff + longer DNS TTL.
_PUMP_BASES = [
    "https://api.pump.fun",
    "https://pumpportal.fun/api",  # fallback community mirror
]
_TIMEOUT = httpx.Timeout(10.0, connect=10.0, read=10.0, write=10.0)
_RETRIES = 6


class PumpFunClient:
    def __init__(self) -> None:
        self._c = httpx.AsyncClient(timeout=_TIMEOUT, headers={"User-Agent": "coinTrader/pump"})

    async def aclose(self) -> None:
        try:
            await self._c.aclose()
        except Exception:
            pass

    async def _aget_json(self, path: str) -> Optional[Dict[str, Any]]:
        last_exc: Optional[Exception] = None
        for i in range(_RETRIES):
            for base in _PUMP_BASES:
                try:
                    r = await self._c.get(base + path)
                    if r.status_code == 200:
                        return r.json()
                except Exception as e:
                    last_exc = e
                    await asyncio.sleep(0.5 * (2**i))
            # rotate bases on backoff
        if last_exc:
            raise last_exc
        return None

    async def trending(self) -> list[dict]:
        data = await self._aget_json("/coins/trending")
        if not data:
            return []
        items = data.get("result") or data.get("data") or data  # tolerate shapes
        return items if isinstance(items, list) else []
