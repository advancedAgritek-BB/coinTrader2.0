from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx

# Public Raydium API endpoints shift occasionally. We try multiple.
_RAYDIUM_ENDPOINTS = [
    "https://api.raydium.io/v2/main/pairs",
    "https://api.raydium.io/pairs",
    "https://api-v3.raydium.io/pools",
]
_TIMEOUT = httpx.Timeout(10.0, connect=10.0, read=10.0, write=10.0)


class RaydiumClient:
    def __init__(self, client: Optional[httpx.Client] = None) -> None:
        self._c = client or httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": "coinTrader/raydium"})

    def close(self) -> None:
        try:
            self._c.close()
        except Exception:
            pass

    def _fetch_json(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            r = self._c.get(url)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None
        return None

    def get_pairs(self) -> list[dict]:
        for base in _RAYDIUM_ENDPOINTS:
            data = self._fetch_json(base)
            if not data:
                continue
            # The shapes differ; normalize to a list of dicts with keys we need.
            if isinstance(data, dict) and "data" in data:
                arr = data["data"]
            elif isinstance(data, list):
                arr = data
            else:
                arr = []
            out = []
            for p in arr:
                out.append(
                    {
                        "address": p.get("ammId") or p.get("id") or p.get("address"),
                        "baseMint": p.get("baseMint") or p.get("base_mint"),
                        "quoteMint": p.get("quoteMint") or p.get("quote_mint"),
                        "liquidityUsd": float(p.get("liquidityUsd") or p.get("liquidity_usd") or 0.0),
                        "volume24hUsd": float(p.get("volume24hUsd") or p.get("volume_24h_usd") or 0.0),
                        "price": float(p.get("price") or p.get("midPrice") or 0.0),
                    }
                )
            if out:
                return out
            time.sleep(0.5)
        return []

    def best_pool_for_mint(self, base_mint: str, *, min_liquidity_usd: float = 5000.0) -> Optional[dict]:
        pools = self.get_pairs()
        candidates = [p for p in pools if (p.get("baseMint") == base_mint or p.get("quoteMint") == base_mint)]
        candidates = [p for p in candidates if p.get("liquidityUsd", 0.0) >= min_liquidity_usd]
        if not candidates:
            return None
        candidates.sort(key=lambda p: (p.get("volume24hUsd", 0.0), p.get("liquidityUsd", 0.0)), reverse=True)
        return candidates[0]
