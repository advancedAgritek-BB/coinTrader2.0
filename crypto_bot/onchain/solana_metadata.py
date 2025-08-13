import os
import logging
from functools import lru_cache
from typing import Dict, List

import requests

FEATURE_ENABLE_HELIUS = os.getenv("FEATURE_ENABLE_HELIUS", "0") in ("1", "true", "True")
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")

logger = logging.getLogger(__name__)
_NOT_FOUND_CACHE: set[str] = set()


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
    if not FEATURE_ENABLE_HELIUS:
        logger.debug("Helius metadata disabled by FEATURE_ENABLE_HELIUS.")
        return {}
    if not HELIUS_API_KEY:
        logger.warning("Helius disabled: HELIUS_API_KEY not set.")
        return {}
    if not mint_addresses:
        return {}

    # Filter out addresses previously marked as not-found
    mint_addresses = [m for m in mint_addresses if m not in _NOT_FOUND_CACHE]
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
        return resp.json() or {}
    except Exception as e:  # pragma: no cover - network / best effort
        logger.warning(
            "Helius metadata fetch failed (%s). Proceeding without metadata.", e
        )
        return {}
