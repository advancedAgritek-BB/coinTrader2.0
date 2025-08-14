import os
import logging
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List

import requests

from crypto_bot.solana.helius_client import HELIUS_API_KEY, helius_available

FEATURE_ENABLE_HELIUS = os.getenv("FEATURE_ENABLE_HELIUS", "0") in ("1", "true", "True")

logger = logging.getLogger(__name__)
_missing_meta_seen = defaultdict(int)
_NOT_FOUND_CACHE: set[str] = set()


def log_missing_metadata(symbol: str) -> None:
    _missing_meta_seen[symbol] += 1
    # Only log the first occurrence and then every 50th to reduce noise
    if _missing_meta_seen[symbol] == 1 or _missing_meta_seen[symbol] % 50 == 0:
        logger.info("No metadata for %s (seen=%d)", symbol, _missing_meta_seen[symbol])


@lru_cache(maxsize=1024)
def _mark_not_found(mint: str) -> bool:
    """Record that ``mint`` was not found and cache the result."""
    _NOT_FOUND_CACHE.add(mint)
    return True


def _do_helius_request(mint_addresses: List[str], *, api_key: str):
    """Perform the Helius metadata request for ``mint_addresses``."""
    url = f"https://api.helius.xyz/v0/token-metadata?api-key={api_key}"
    payload = {"mintAccounts": mint_addresses}
    return requests.post(url, json=payload, timeout=10)


def fetch_token_metadata(mint_addresses: List[str]) -> Dict[str, Dict]:
    """Return token metadata for ``mint_addresses`` via Helius."""
    if not mint_addresses:
        return {}
    if not FEATURE_ENABLE_HELIUS or not helius_available:
        return {m: {"metadata_unknown": True} for m in mint_addresses if m}

    # Filter out addresses previously marked as not-found and blank entries
    mint_addresses = [m for m in mint_addresses if m and m not in _NOT_FOUND_CACHE]
    if not mint_addresses:
        return {}

    try:
        resp = _do_helius_request(mint_addresses, api_key=HELIUS_API_KEY)
        if 400 <= resp.status_code < 500:
            logger.info(
                "Helius returned %s for %d mints; disabling further lookups for these mints.",
                resp.status_code,
                len(mint_addresses),
            )
            for m in mint_addresses:
                _mark_not_found(m)
            return {}
        resp.raise_for_status()
        data = resp.json() or []
    except Exception as e:  # pragma: no cover - network / best effort
        logger.warning(
            "Helius metadata fetch failed (%s). Proceeding without metadata.", e
        )
        return {m: {"metadata_unknown": True} for m in mint_addresses}

    items = data if isinstance(data, list) else data.get("tokens") or data.get("data") or []
    if isinstance(items, dict):
        items = list(items.values())
    result: Dict[str, Dict] = {}
    for item in items if isinstance(items, list) else []:
        mint = (
            item.get("onChainAccountInfo", {}).get("mint")
            or item.get("mint")
            or item.get("address")
            or item.get("tokenMint")
        )
        if isinstance(mint, str):
            result[mint] = item
        else:
            log_missing_metadata(str(mint))
    return result
