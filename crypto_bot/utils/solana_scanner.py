from __future__ import annotations

"""Fetch newly launched Solana tokens from public APIs."""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List

import aiohttp
import ccxt.async_support as ccxt

from .gecko import gecko_request
from . import symbol_scoring
from . import kraken as kraken_utils

logger = logging.getLogger(__name__)

# Global min volume filter updated by ``get_solana_new_tokens``
_MIN_VOLUME_USD = 0.0

RAYDIUM_URL = "https://api.raydium.io/pairs"
PUMP_FUN_URL = "https://api.pump.fun/tokens"

# Timestamp of the most recent Pump.fun token processed
last_pump_ts: float = 0.0

# Persist last processed Raydium creation timestamp
RAYDIUM_TS_FILE = (
    Path(__file__).resolve().parents[2] / "cache" / "raydium_last_ts.txt"
)
try:
    _LAST_RAYDIUM_TS = float(RAYDIUM_TS_FILE.read_text())
except Exception:
    _LAST_RAYDIUM_TS = 0.0


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


# Cached exchange instance
_EXCHANGE = None


def _get_exchange(config: dict):
    """Return a cached ccxt exchange instance."""

    global _EXCHANGE
    if _EXCHANGE is None:
        ex = config.get("exchange", "kraken")
        if isinstance(ex, dict):
            ex_name = ex.get("name", "kraken").lower()
            params = {"enableRateLimit": True}
            timeout = ex.get("request_timeout_ms")
            if timeout:
                params["timeout"] = int(timeout)
            max_conc = ex.get("max_concurrency")
        else:
            ex_name = str(ex).lower()
            params = {"enableRateLimit": True}
            max_conc = None

        if ex_name == "kraken":
            _EXCHANGE = kraken_utils.get_client()
        else:
            exchange_cls = getattr(ccxt, ex_name)
            _EXCHANGE = exchange_cls(params)
            if max_conc is not None:
                setattr(_EXCHANGE, "max_concurrency", int(max_conc))
    return _EXCHANGE


async def _extract_tokens(data: list | dict) -> List[str]:
    """Return new Raydium pool mints from ``data``.

    The v2 API returns a list of pair objects with ``base.address`` providing
    the token mint.  Results are filtered to only include pools where:

    * ``liquidity_locked`` is ``True``
    * ``creationTimestamp`` is newer than the last poll
    * ``volume24h`` >= 100_000 and ``liquidity`` >= 50_000

    The most recent ``creationTimestamp`` encountered is persisted to disk so
    subsequent polls skip older entries.
    """

    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    global _LAST_RAYDIUM_TS
    latest = _LAST_RAYDIUM_TS
    results: list[tuple[str, float]] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        base = item.get("base") or {}
        mint = base.get("address") if isinstance(base, dict) else None
        try:
            liq = float(item.get("liquidity") or 0)
            vol = float(item.get("volume24h") or 0)
            ts = float(item.get("creationTimestamp") or 0)
            locked = bool(item.get("liquidity_locked"))
        except Exception:
            continue
        if (
            not mint
            or not locked
            or ts <= _LAST_RAYDIUM_TS
            or vol < 100_000
            or liq < 50_000
        ):
            continue
        results.append((str(mint), ts))
        if ts > latest:
            latest = ts

    if results and latest > _LAST_RAYDIUM_TS:
        _LAST_RAYDIUM_TS = latest
        try:
            RAYDIUM_TS_FILE.parent.mkdir(parents=True, exist_ok=True)
            RAYDIUM_TS_FILE.write_text(str(int(latest)))
        except Exception:  # pragma: no cover - best effort
            pass

    results.sort(key=lambda x: x[1], reverse=True)
    return [mint for mint, _ in results]


async def _extract_pump_fun_tokens(data: list | dict) -> List[str]:
    """Return token mints from Pump.fun responses."""

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
            results.append(str(mint))
    return results


async def fetch_new_raydium_pools(limit: int) -> List[str]:
    """Return new Raydium pool token mints."""

    url = f"{RAYDIUM_URL}?limit={limit}"
    data = await _fetch_json(url)
    if not data:
        return []

    items = data.get("data") if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        return []

    filtered: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mint = item.get("base", {}).get("address")
        if not isinstance(mint, str):
            continue
        liquidity = item.get("liquidity") or 0.0
        try:
            liquidity = float(liquidity)
        except Exception:
            liquidity = 0.0
        creation = item.get("creation_timestamp") or 0.0
        try:
            creation = float(creation)
        except Exception:
            creation = 0.0
        if not item.get("liquidity_locked") or liquidity < _MIN_VOLUME_USD or creation <= 0:
            continue
        vol = item.get("volume24h") or 0.0
        filtered.append({"tokenMint": mint, "volumeUsd": vol})

    tokens = await _extract_tokens({"data": filtered})
    return tokens[:limit]


async def fetch_pump_fun_launches(*args) -> List[str]:
    """Return recent Pump.fun launches.

    The function accepts ``limit`` as the last positional argument to
    remain backward compatible with older call sites that passed an API key
    first.  The Pump.fun API returns a JSON array of token objects.  Each
    item must provide ``created_at``, ``initial_buy``, ``market_cap`` and
    ``twitter`` fields.  Only tokens newer than ``last_pump_ts`` are
    returned.
    """

    if not args:
        return []
    limit = int(args[-1])

    global last_pump_ts

    url = f"{PUMP_FUN_URL}?limit={limit}&offset=0"
    data = await _fetch_json(url)
    if not isinstance(data, list):
        return []

    results: List[str] = []
    max_ts = last_pump_ts

    for item in data:
        if len(results) >= limit:
            break
        if not isinstance(item, dict):
            continue

        created = item.get("created_at") or item.get("createdAt")
        initial_buy = item.get("initial_buy") or item.get("initialBuy")
        market_cap = item.get("market_cap") or item.get("marketCap")
        twitter = item.get("twitter") or item.get("twitter_profile")
        if not (created and initial_buy and market_cap and twitter):
            continue

        try:
            ts_val = datetime.fromisoformat(str(created).replace("Z", "+00:00")).timestamp()
            mcap = float(market_cap)
            if mcap <= 0:
                continue
        except Exception:
            continue
        if not bool(initial_buy) or ts_val <= last_pump_ts:
            continue

        if ts_val > max_ts:
            max_ts = ts_val

        mint = (
            item.get("mint")
            or item.get("tokenMint")
            or item.get("token_mint")
            or item.get("address")
        )
        if mint:
            results.append(str(mint))

    last_pump_ts = max_ts
    return results


async def get_solana_new_tokens(config: dict, exchange=None) -> List[str]:
    """Return deduplicated Solana token symbols from multiple sources."""

    global _MIN_VOLUME_USD

    limit = int(config.get("max_tokens_per_scan", 0)) or 20
    _MIN_VOLUME_USD = float(config.get("min_volume_usd", 0.0))
    pump_key = str(config.get("pump_fun_api_key", ""))
    raydium_key = str(config.get("raydium_api_key", ""))
    gecko_search = bool(config.get("gecko_search", True))

    tasks: list[asyncio.Future] = []

    if raydium_key:
        coro = fetch_new_raydium_pools(limit)
        if not asyncio.iscoroutine(coro):
            async def _wrap(res=coro):
                return res
            coro = _wrap()
        tasks.append(coro)

    if pump_key or raydium_key:
        coro = fetch_pump_fun_launches(limit)
        if not asyncio.iscoroutine(coro):
            async def _wrap(res=coro):
                return res
            coro = _wrap()
        tasks.append(coro)

    if not tasks:
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
    if exchange is None:
        exchange = _get_exchange(config)

    scores = await asyncio.gather(
        *[
            symbol_scoring.score_symbol(
                exchange, sym, vol, 0.0, 0.0, 1.0, config
            )
            for sym, vol in final
        ]
    )

    scored = [
        (sym, score)
        for (sym, _), score in zip(final, scores)
        if score >= min_score
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in scored]
