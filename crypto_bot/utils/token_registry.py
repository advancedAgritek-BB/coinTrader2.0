from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict

import aiohttp

logger = logging.getLogger(__name__)

TOKEN_REGISTRY_URL = (
    "https://raw.githubusercontent.com/solana-labs/token-list/main/src/tokens/solana.tokenlist.json"
)

CACHE_FILE = Path(__file__).resolve().parents[2] / "cache" / "token_mints.json"

# Mapping of token symbols to Solana mints. ``load_token_mints`` populates this
# dictionary at runtime.
TOKEN_MINTS: Dict[str, str] = {}

_LOADED = False


async def load_token_mints(url: str | None = None) -> Dict[str, str]:
    """Return mapping of token symbols to mint addresses.

    The list is fetched from ``url`` or ``TOKEN_MINTS_URL`` environment variable.
    Results are cached on disk and subsequent calls return an empty dict.
    """
    global _LOADED
    if _LOADED:
        return {}

    fetch_url = url or os.getenv("TOKEN_MINTS_URL", TOKEN_REGISTRY_URL)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(fetch_url, timeout=10) as resp:
                resp.raise_for_status()
                # Allow JSON served with incorrect Content-Type like text/plain
                data = await resp.json(content_type=None)
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Failed to fetch token registry: %s", exc)
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE) as f:
                    cached = json.load(f)
                if isinstance(cached, dict):
                    _LOADED = True
                    return {str(k): str(v) for k, v in cached.items()}
            except Exception as err:  # pragma: no cover - best effort
                logger.error("Failed to read cache: %s", err)
        return {}

    tokens = data.get("tokens") or data.get("data", {}).get("tokens") or []
    result: Dict[str, str] = {}
    for item in tokens:
        symbol = item.get("symbol") or item.get("ticker")
        mint = item.get("address") or item.get("mint") or item.get("tokenMint")
        if isinstance(symbol, str) and isinstance(mint, str):
            result[symbol.upper()] = mint
    if result:
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as exc:  # pragma: no cover - optional cache
            logger.error("Failed to write %s: %s", CACHE_FILE, exc)
    _LOADED = True
    return result


def _write_cache() -> None:
    """Write ``TOKEN_MINTS`` to :data:`CACHE_FILE`."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(TOKEN_MINTS, f, indent=2)
    except Exception as exc:  # pragma: no cover - optional cache
        logger.error("Failed to write %s: %s", CACHE_FILE, exc)


def set_token_mints(mapping: dict[str, str]) -> None:
    """Replace ``TOKEN_MINTS`` with ``mapping`` after normalizing keys."""
    TOKEN_MINTS.clear()
    TOKEN_MINTS.update({k.upper(): v for k, v in mapping.items()})
    _write_cache()


async def get_mint_from_gecko(base: str) -> str | None:
    """Return Solana mint address for ``base`` using GeckoTerminal.

    ``None`` is returned if the request fails or no token matches the
    given symbol.
    """

    from urllib.parse import quote_plus

    url = (
        "https://api.geckoterminal.com/api/v2/search/tokens"
        f"?query={quote_plus(str(base))}&network=solana"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Gecko lookup failed for %s: %s", base, exc)
        return None

    items = data.get("data") if isinstance(data, dict) else []
    if not isinstance(items, list) or not items:
        return None

    item = items[0]
    attrs = item.get("attributes", {}) if isinstance(item, dict) else {}
    mint = attrs.get("address") or item.get("id")
    return str(mint) if isinstance(mint, str) else None
