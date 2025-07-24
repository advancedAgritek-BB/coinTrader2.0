from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List

import aiohttp

from .gecko import gecko_request
logger = logging.getLogger(__name__)

TOKEN_REGISTRY_URL = (
    "https://raw.githubusercontent.com/solana-labs/token-list/main/src/tokens/solana.tokenlist.json"
)

# Primary token list from Jupiter API
# ``station.jup.ag`` now redirects to ``dev.jup.ag`` which returns ``404``.
# The latest stable token list is hosted at ``https://token.jup.ag/all``.
JUPITER_TOKEN_URL = "https://token.jup.ag/all"

# Batch metadata endpoint for resolving unknown symbols
HELIUS_TOKEN_API = "https://api.helius.xyz/v0/tokens/metadata"

CACHE_FILE = Path(__file__).resolve().parents[2] / "cache" / "token_mints.json"

# Mapping of token symbols to Solana mints. ``load_token_mints`` populates this
# dictionary at runtime.
TOKEN_MINTS: Dict[str, str] = {}

_LOADED = False

async def fetch_from_jupiter() -> Dict[str, str]:
    """Return mapping of symbols to mints using Jupiter token list."""
    async with aiohttp.ClientSession() as session:
        async with session.get(JUPITER_TOKEN_URL, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

    result: Dict[str, str] = {}
    tokens = data if isinstance(data, list) else data.get("tokens") or []
    for item in tokens:
        symbol = item.get("symbol") or item.get("ticker")
        mint = item.get("address") or item.get("mint") or item.get("tokenMint")
        if isinstance(symbol, str) and isinstance(mint, str):
            result[symbol.upper()] = mint
    return result


async def load_token_mints(
    url: str | None = None,
    *,
    unknown: List[str] | None = None,
    force_refresh: bool = False,
) -> Dict[str, str]:
    """Return mapping of token symbols to mint addresses.

    The list is fetched from ``url`` or ``TOKEN_MINTS_URL`` environment variable.
    Results are cached on disk and subsequent calls return an empty dict unless
    ``force_refresh`` is ``True``.
    The Solana list is fetched from Jupiter first then GitHub as a fallback.
    Cached results are reused unless ``force_refresh`` is ``True``.
    Unknown ``symbols`` can be resolved via the Helius API.
    """
    global _LOADED
    if _LOADED and not force_refresh:
        return {}

    mapping: Dict[str, str] = {}

    try:
        mapping.update(await fetch_from_jupiter())
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Failed to fetch Jupiter tokens: %s", exc)

    if not mapping:
        fetch_url = url or os.getenv("TOKEN_MINTS_URL", TOKEN_REGISTRY_URL)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(fetch_url, timeout=10) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
            tokens = data.get("tokens") or data.get("data", {}).get("tokens") or []
            for item in tokens:
                symbol = item.get("symbol") or item.get("ticker")
                mint = (
                    item.get("address") or item.get("mint") or item.get("tokenMint")
                )
                if isinstance(symbol, str) and isinstance(mint, str):
                    mapping[symbol.upper()] = mint
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch token registry: %s", exc)

    if not mapping and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cached = json.load(f)
            if isinstance(cached, dict):
                mapping.update({str(k).upper(): str(v) for k, v in cached.items()})
        except Exception as err:  # pragma: no cover - best effort
            logger.error("Failed to read cache: %s", err)

    if unknown:
        try:
            mapping.update(await fetch_from_helius(unknown))
        except Exception as exc:  # pragma: no cover - network failures
            logger.error("Failed to fetch Helius metadata: %s", exc)

    if mapping:
        TOKEN_MINTS.update({k.upper(): v for k, v in mapping.items()})
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(TOKEN_MINTS, f, indent=2)
        except Exception as exc:  # pragma: no cover - optional cache
            logger.error("Failed to write %s: %s", CACHE_FILE, exc)

    _LOADED = True
    return mapping


def _write_cache() -> None:
    """Write ``TOKEN_MINTS`` to :data:`CACHE_FILE`."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(TOKEN_MINTS, f, indent=2)
    except Exception as exc:  # pragma: no cover - optional cache
        logger.error("Failed to write %s: %s", CACHE_FILE, exc)


# Additional mints discovered via manual searches
TOKEN_MINTS.update({
    "AI16Z": "HeLp6NuQkmYB4pYWo2zYs22mESHXPQYzXbB8n4V98jwC",
    "BERA": "A7y2wgyytufsxjg2ub616zqnte3x62f7fcp8fujdmoon",
    "EUROP": "pD6L7wWeei1LJqb7tmnpfEnvkcBvqMgkfqvg23Bpump",
    "FARTCOIN": "Bzc9NZfMqkXR6fz1DBph7BDf9BroyEf6pnzESP7v5iiw",
    "RLUSD": "BkbjmJVa84eiGyp27FTofuQVFLqmKFev4ZPZ3U33pump",
    "USDG": "2gc4f72GkEtggrkUDJRSbLcBpEUPPPFsnDGJJeNKpump",  # Assuming Unlimited Solana Dump
    "VIRTUAL": "2FupRnaRfnyPHg798WsCBMGAauEkrhMs4YN7nBmujPtM",
    "XMR": "Fi9GeixxfhMEGfnAe75nJVrwPqfVefyS6fgmyiTxkS6q",  # Wrapped, verify
    "MELANIA": "FUAfBo2jgks6gB4Z4LfZkqSZgzNucisEHqnNebaRxM1P",
    "PENGU": "2zMMhcVQEXDtdE6vsFS7S7D5oUodfJHE8vd1gnBouauv",
    "USDR": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT as proxy
    "USTC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC proxy (adjust if needed)
    "TRUMP": "6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN",
    # Add more as needed; skip USDQ/USTC/XTZ as non-Solana
})
_write_cache()  # Save immediately


async def refresh_mints() -> None:
    """Force refresh cached token mints and add known symbols."""
    await load_token_mints(
        force_refresh=True,
        unknown=[
            "AI16Z",
            "FARTCOIN",
            "MELANIA",
            "PENGU",
            "RLUSD",
            "VIRTUAL",
            "USDG",
            "USDR",
            "USTC",
            "TRUMP",
        ],
    )
    TOKEN_MINTS.update(
        {
            "AI16Z": "HeLp6NuQkmYB4pYWo2zYs22mESHXPQYzXbB8n4V98jwC",
            "FARTCOIN": "Bzc9NZfMqkXR6fz1DBph7BDf9BroyEf6pnzESP7v5iiw",
            "MELANIA": "FUAfBo2jgks6gB4Z4LfZkqSZgzNucisEHqnNebaRxM1P",
            "PENGU": "2zMMhcVQEXDtdE6vsFS7S7D5oUodfJHE8vd1gnBouauv",
            "RLUSD": "BkbjmJVa84eiGyp27FTofuQVFLqmKFev4ZPZ3U33pump",
            "VIRTUAL": "2FupRnaRfnyPHg798WsCBMGAauEkrhMs4YN7nBmujPtM",
            "USDG": "2u1tszSeqZ3qBWF3uNGPFc8TzMk2tdiwknnRMWGWjGWH",
            "USDR": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
            "USTC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "TRUMP": "6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN",
        }
    )
    _write_cache()
    logger.info("Refreshed TOKEN_MINTS with %d entries", len(TOKEN_MINTS))


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

    mint = None
    try:
        data = await gecko_request(url)
    except Exception as exc:  # pragma: no cover - network
        logger.error("Gecko lookup failed: %s", exc)
        data = None

    if isinstance(data, dict):
        items = data.get("data")
        if isinstance(items, list) and items:
            item = items[0]
            attrs = item.get("attributes", {}) if isinstance(item, dict) else {}
            mint = attrs.get("address") or item.get("id")
            if isinstance(mint, str):
                return mint

    # Fallback: attempt to search DexScreener if Gecko fails
    fallback_url = f"https://dexscreener.com/solana?query={quote_plus(str(base))}"
    try:
        data = await gecko_request(fallback_url)
    except Exception as exc:  # pragma: no cover - network
        logger.error("Dexscreener lookup failed: %s", exc)
        data = None
    if (
        isinstance(data, dict)
        and data.get("pairs")
        and isinstance(data["pairs"], list)
    ):
        return data["pairs"][0].get("baseToken", {}).get("address")

    # New fallback: query Helius if GeckoTerminal and Dexscreener fail
    try:
        helius = await fetch_from_helius([base])
    except Exception:  # pragma: no cover - network failures
        helius = {}
    return helius.get(base.upper())


async def fetch_from_helius(symbols: Iterable[str]) -> Dict[str, str]:
    """Return mapping of ``symbols`` to mint addresses using Helius.

    Parameters
    ----------
    symbols:
        Iterable of token symbols to resolve.
    """

    api_key = os.getenv("HELIUS_KEY", "")
    if not symbols:
        return {}
    url = (
        "https://api.helius.xyz/v0/tokens/metadata"
        f"?symbols={','.join(symbols)}"
        + (f"&api-key={api_key}" if api_key else "")
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except aiohttp.ClientError as exc:  # pragma: no cover - network
        logger.error("Helius lookup failed: %s", exc)
        return {}
    except Exception as exc:  # pragma: no cover - network
        logger.error("Helius lookup error: %s", exc)
        return {}

    result: Dict[str, str] = {}
    if isinstance(data, list):
        items = data
    else:
        items = data.get("tokens") or data.get("data") or []
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        return {}
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol") or item.get("ticker")
        mint = item.get("mint") or item.get("address") or item.get("tokenMint")
        if isinstance(symbol, str) and isinstance(mint, str):
            result[symbol.upper()] = mint
    return result
