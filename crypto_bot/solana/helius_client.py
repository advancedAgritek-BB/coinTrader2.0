from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

HELIUS_API_KEY = (
    os.getenv("HELIUS_API_KEY")
    or os.getenv("HELIUS_KEY")
    or os.getenv("HELIUS")
    or ""
)

_HELIUS_BASE = "https://api.helius.xyz"
_TIMEOUT = httpx.Timeout(10.0, connect=10.0, read=10.0, write=10.0)
_RETRIES = 3
_BACKOFF = 0.75


@dataclass
class TokenMetadata:
    mint: str
    symbol: Optional[str] = None
    name: Optional[str] = None
    decimals: Optional[int] = None
    image: Optional[str] = None
    freeze_authority: Optional[str] = None
    mint_authority: Optional[str] = None
    program: Optional[str] = None


def helius_available() -> bool:
    """
    True if we have a key AND Helius responds to a trivial request.
    We avoid false negatives by retrying transient network errors.
    """
    if not HELIUS_API_KEY:
        return False
    url = f"{_HELIUS_BASE}/v0/addresses?api-key={HELIUS_API_KEY}"
    for i in range(_RETRIES):
        try:
            r = httpx.get(url, timeout=_TIMEOUT)
            if r.status_code in (200, 404, 400):  # 404/400 still proves reachability
                return True
        except Exception:
            time.sleep(_BACKOFF * (2**i))
    return False


class HeliusClient:
    def __init__(self, api_key: Optional[str] = None, *, client: Optional[httpx.Client] = None) -> None:
        self.api_key = api_key or HELIUS_API_KEY
        self._client = client or httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": "coinTrader/helius"})
        if not self.api_key:
            raise RuntimeError("HELIUS_API_KEY missing in environment.")

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def _url(self, path: str) -> str:
        return f"{_HELIUS_BASE}{path}?api-key={self.api_key}"

    def get_token_metadata(self, mint: str) -> Optional[TokenMetadata]:
        """
        Helius token metadata API: /v0/token-metadata?mint=...
        Fallback to /v0/tokens/metadata if needed.
        """
        # Primary
        url = self._url("/v0/token-metadata") + f"&mint={mint}"
        r = self._client.get(url)
        if r.status_code == 200:
            try:
                data = r.json() or {}
                return _parse_token_metadata(mint, data)
            except Exception:
                pass

        # Fallback (batch style endpoint)
        url2 = self._url("/v0/tokens/metadata")
        r2 = self._client.post(url2, json={"mintAccounts": [mint]})
        if r2.status_code == 200:
            try:
                arr = r2.json() or []
                if arr:
                    return _parse_token_metadata(mint, arr[0])
            except Exception:
                pass
        return None


def _parse_token_metadata(mint: str, payload: Dict[str, Any]) -> TokenMetadata:
    attrs = payload.get("onChainMetadata", {}).get("metadata", {}).get("data", {}).get("attrs", {})
    symbol = payload.get("symbol") or payload.get("token", {}).get("symbol")
    name = payload.get("name") or payload.get("token", {}).get("name")
    decimals = payload.get("decimals") or payload.get("token", {}).get("decimals")
    image = payload.get("image")
    freeze_authority = attrs.get("freezeAuthority") or payload.get("freezeAuthority")
    mint_authority = attrs.get("mintAuthority") or payload.get("mintAuthority")
    program = payload.get("program") or payload.get("programId")
    return TokenMetadata(
        mint=mint,
        symbol=symbol,
        name=name,
        decimals=decimals,
        image=image,
        freeze_authority=freeze_authority,
        mint_authority=mint_authority,
        program=program,
    )

