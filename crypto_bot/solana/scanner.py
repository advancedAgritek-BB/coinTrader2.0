from __future__ import annotations

"""Utilities for scanning new Solana tokens."""

import asyncio
import os
from typing import Mapping, List

import aiohttp
import ccxt.async_support as ccxt

from crypto_bot.utils.solana_scanner import search_geckoterminal_token
from crypto_bot.utils import symbol_scoring
from crypto_bot.utils import kraken as kraken_utils
from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "meme_sniper.log")


_EXCHANGE: kraken_utils.KrakenClient | None = None


def _get_exchange(cfg: Mapping[str, object]):
    """Return a cached exchange instance based on ``cfg`` options."""

    global _EXCHANGE
    if _EXCHANGE is None:
        ex_cfg = cfg.get("exchange", "kraken")
        if isinstance(ex_cfg, dict):
            ex_name = ex_cfg.get("name", "kraken").lower()
            params = {"enableRateLimit": True}
            timeout = ex_cfg.get("request_timeout_ms")
            if timeout:
                params["timeout"] = int(timeout)
            max_conc = ex_cfg.get("max_concurrency")
        else:
            ex_name = str(ex_cfg).lower()
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


async def get_solana_new_tokens(
    cfg: Mapping[str, object], exchange: kraken_utils.KrakenClient | None = None
) -> List[str]:
    """Return a list of new token mint addresses using ``cfg`` options."""

    url = str(cfg.get("url", ""))
    if not url:
        return []

    key = os.getenv("HELIUS_KEY", "")
    if "${HELIUS_KEY}" in url:
        url = url.replace("${HELIUS_KEY}", key)
    if "YOUR_KEY" in url:
        url = url.replace("YOUR_KEY", key)

    limit = int(cfg.get("limit", 0))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        logger.error("Solana scanner error: %s", exc)
        return []

    tokens = data.get("tokens") or data.get("mints") or data
    results: List[str] = []
    if isinstance(tokens, list):
        for item in tokens:
            if isinstance(item, str):
                results.append(item)
            elif isinstance(item, Mapping):
                mint = item.get("mint") or item.get("tokenMint") or item.get("token_mint")
                if mint:
                    results.append(str(mint))
    elif isinstance(tokens, Mapping):
        for mint in tokens.values():
            if isinstance(mint, str):
                results.append(mint)
    if limit:
        results = results[:limit]

    if not results:
        return []

    search_results = await asyncio.gather(
        *[search_geckoterminal_token(m) for m in results]
    )

    resolved: list[tuple[str, float]] = []
    seen: set[str] = set()
    for res in search_results:
        if not res:
            continue
        mint, vol = res
        sym = f"{mint}/USDC"
        if sym not in seen:
            seen.add(sym)
            resolved.append((sym, vol))

    if not resolved:
        return []

    min_score = float(cfg.get("min_symbol_score", 0.0))
    if exchange is None:
        exchange = _get_exchange(cfg)

    scores = await asyncio.gather(
        *[
            symbol_scoring.score_symbol(
                exchange, sym, vol, 0.0, 0.0, 1.0, cfg
            )
            for sym, vol in resolved
        ]
    )

    scored = [
        (sym, score) for (sym, _), score in zip(resolved, scores) if score >= min_score
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in scored]
