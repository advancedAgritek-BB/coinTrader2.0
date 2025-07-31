from __future__ import annotations

"""Fetch newly launched Solana tokens from public APIs."""

import asyncio
import logging
import os
from typing import List
import aiohttp
import ccxt.async_support as ccxt
from .gecko import gecko_request

from .token_registry import TOKEN_MINTS, get_mint_from_gecko, fetch_from_helius

from . import symbol_scoring

logger = logging.getLogger(__name__)

# Global min volume filter updated by ``get_solana_new_tokens``
_MIN_VOLUME_USD = 0.0

RAYDIUM_URL = "https://api.raydium.io/pairs"
PUMP_FUN_URL = "https://client-api.prod.pump.fun/v1/launches"


async def search_geckoterminal_token(query: str) -> tuple[str, float] | None:
    """Return ``(mint, volume)`` from GeckoTerminal token search.

    The function queries ``/api/v2/search/pools`` with ``query`` and
    ``network=solana`` and returns the first result's base token mint
    and 24h volume in USD. ``None`` is returned when the request fails
    or no results are available.
    """

    from urllib.parse import quote_plus

    url = (
        "https://api.geckoterminal.com/api/v2/search/pools"
        f"?query={quote_plus(query)}&network=solana"
    )

    data = await gecko_request(url)
    if not data:
        return None

    items = data.get("data") if isinstance(data, dict) else []
    if not isinstance(items, list) or not items:
        return None

    item = items[0] if isinstance(items[0], dict) else None
    if not item:
        return None

    mint = (
        item.get("relationships", {})
        .get("base_token", {})
        .get("data", {})
        .get("id")
    )
    if not isinstance(mint, str):
        return None
    if mint.startswith("solana_"):
        mint = mint[len("solana_") :]

    attrs = item.get("attributes", {})
    try:
        volume = float(attrs.get("volume_usd_h24") or 0.0)
    except Exception:
        volume = 0.0

    return mint, volume


async def _fetch_json(url: str) -> list | dict | None:
    """Return parsed JSON from ``url`` using ``aiohttp``."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        logger.error("Solana scanner request failed: %s", exc)
        return None


async def _close_exchange(exchange) -> None:
    """Close ``exchange`` ignoring errors."""
    close = getattr(exchange, "close", None)
    if close:
        try:
            if asyncio.iscoroutinefunction(close):
                await close()
            else:
                close()
        except Exception:  # pragma: no cover - best effort
            pass


async def _download_snapshot(mint: str, bucket: str):
    """Return a snapshot dataframe for ``mint`` from Supabase.

    The function downloads ``f"{mint}.parquet"`` from ``bucket`` using
    ``SUPABASE_URL`` and ``SUPABASE_KEY`` environment variables.  ``None`` is
    returned when the download or parsing fails.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        logger.error("Missing Supabase credentials")
        return None

    try:  # pragma: no cover - optional dependency
        from supabase import create_client
        import pandas as pd
    except Exception as exc:  # pragma: no cover - log import failure
        logger.error("Supabase client unavailable: %s", exc)
        return None

    try:
        client = create_client(url, key)
        data = client.storage.from_(bucket).download(f"{mint}.parquet")
        from io import BytesIO

        return pd.read_parquet(BytesIO(data))
    except Exception as exc:
        logger.error("Failed to download snapshot for %s: %s", mint, exc)
        return None


async def _extract_tokens(data: list | dict) -> List[str]:
    """Return token mints from ``data`` respecting ``_MIN_VOLUME_USD``.

    Tokens not present in :data:`TOKEN_MINTS` are verified via
    :func:`get_mint_from_gecko`. Unresolvable entries are skipped.
    """
    items = data.get("data") if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        return []

    results: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mint = (
            item.get("tokenMint")
            or item.get("token_mint")
            or item.get("mint")
            or item.get("address")
        )
        if not mint:
            continue
        vol = (
            item.get("volumeUsd")
            or item.get("volume_usd")
            or item.get("liquidityUsd")
            or item.get("liquidity_usd")
            or 0.0
        )
        try:
            volume = float(vol)
        except Exception:
            volume = 0.0
        if volume >= _MIN_VOLUME_USD:
            base = str(mint).split("/")[0]
            if base not in TOKEN_MINTS:
                resolved = await get_mint_from_gecko(base)
                if not resolved:
                    helius = await fetch_from_helius([base])
                    resolved = helius.get(base.upper()) if helius else None
                if resolved:
                    TOKEN_MINTS[base] = resolved
                else:
                    logger.warning("Mint lookup failed for %s", base)
                    continue
            results.append(str(mint))
    return results


async def fetch_new_raydium_pools(api_key: str, limit: int) -> List[str]:
    """Return new Raydium pool token mints."""
    url = f"{RAYDIUM_URL}?apiKey={api_key}&limit={limit}"
    data = await _fetch_json(url)
    if data is None:
        logger.warning("Failed to fetch Raydium pools")
        return []
    tokens = await _extract_tokens(data)
    return tokens[:limit]


async def fetch_pump_fun_launches(api_key: str, limit: int) -> List[str]:
    """Return recent Pump.fun launches."""
    url = f"{PUMP_FUN_URL}?api-key={api_key}&limit={limit}"
    data = await _fetch_json(url)
    if data is None:
        logger.warning("Failed to fetch Pump.fun launches")
        return []
    tokens = await _extract_tokens(data)
    return tokens[:limit]


async def get_solana_new_tokens(config: dict) -> List[str]:
    """Return deduplicated Solana token symbols from multiple sources."""

    global _MIN_VOLUME_USD

    limit = int(config.get("max_tokens_per_scan", 0)) or 20
    _MIN_VOLUME_USD = float(config.get("min_volume_usd", 0.0))
    raydium_key = str(config.get("raydium_api_key", ""))
    pump_key = str(config.get("pump_fun_api_key", ""))
    gecko_search = bool(config.get("gecko_search", True))

    if not raydium_key:
        logger.warning("raydium_api_key not set; skipping Raydium pools")
    if not pump_key:
        logger.warning("pump_fun_api_key not set; skipping Pump.fun launches")

    tasks = []
    if raydium_key:
        coro = fetch_new_raydium_pools(raydium_key, limit)
        if not asyncio.iscoroutine(coro):
            async def _wrap(res=coro):
                return res
            coro = _wrap()
        tasks.append(coro)
    if pump_key:
        coro = fetch_pump_fun_launches(pump_key, limit)
        if not asyncio.iscoroutine(coro):
            async def _wrap(res=coro):
                return res
            coro = _wrap()
        tasks.append(coro)

    if not tasks:
        logger.warning("No Solana API keys provided; skipping token scan")
        return []

    results = await asyncio.gather(*tasks)
    candidates: list[str] = []
    seen_raw: set[str] = set()
    for res in results:
        for mint in res:
            if mint not in seen_raw:
                seen_raw.add(mint)
                candidates.append(mint)
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break

    if not gecko_search:
        return [f"{m}/USDC" for m in candidates]

    search_results = await asyncio.gather(
        *[search_geckoterminal_token(m) for m in candidates]
    )

    final: list[tuple[str, float]] = []
    seen: set[str] = set()
    for res in search_results:
        if not res:
            continue
        mint, vol = res
        if vol >= _MIN_VOLUME_USD and mint not in seen:
            seen.add(mint)
            final.append((f"{mint}/USDC", vol))
        if len(final) >= limit:
            break

    if not final:
        return []

    min_score = float(config.get("min_symbol_score", 0.0))
    ex_name = str(config.get("exchange", "kraken")).lower()
    exchange_cls = getattr(ccxt, ex_name)
    exchange = exchange_cls({"enableRateLimit": True})

    try:
        scores = await asyncio.gather(
            *[
                symbol_scoring.score_symbol(
                    exchange, sym, vol, 0.0, 0.0, 1.0, config
                )
                for sym, vol in final
            ]
        )
    finally:
        await _close_exchange(exchange)

    scored = [
        (sym, score)
        for (sym, _), score in zip(final, scores)
        if score >= min_score
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored:
        return []

    if config.get("ml_filter"):
        bucket = str(config.get("supabase_bucket", "snapshots"))
        snapshots = await asyncio.gather(
            *[
                _download_snapshot(sym.split("/")[0], bucket)
                for sym, _ in scored
            ]
        )

        try:  # pragma: no cover - optional dependency
            import regime_lgbm
        except Exception:
            probs = [1.0 for _ in snapshots]
        else:
            probs = []
            for snap in snapshots:
                try:
                    prob = float(regime_lgbm.predict(snap))
                except Exception:
                    prob = 0.0
                probs.append(prob)

        filtered = [
            (sym, score, prob)
            for (sym, score), prob in zip(scored, probs)
            if prob >= 0.5
        ]

        filtered.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [sym for sym, _, _ in filtered]

    return [sym for sym, _ in scored]
