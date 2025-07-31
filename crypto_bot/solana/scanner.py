from __future__ import annotations

"""Utilities for scanning new Solana tokens using :class:`PoolWatcher`."""

import asyncio
import logging
from typing import Mapping, List
import aiohttp

from .watcher import PoolWatcher
from crypto_bot.utils.token_registry import TOKEN_MINTS
from crypto_bot.utils.solana_scanner import search_geckoterminal_token
from crypto_bot.utils import symbol_scoring
import ccxt

logger = logging.getLogger(__name__)


async def get_solana_new_tokens(
    config: Mapping[str, object], timeout_seconds: float | None = None
) -> List[str]:
    """Return new Solana token mints using ``config`` options.

    Parameters
    ----------
    config:
        Configuration mapping. ``config["solana_scanner"]`` is used when
        available to retrieve the watcher settings.
    """

    sol_cfg = config.get("solana_scanner") if "solana_scanner" in config else config
    if not isinstance(sol_cfg, Mapping):
        return []

    max_tokens = int(sol_cfg.get("max_tokens_per_scan", 200))
    if max_tokens <= 0:
        return []
    timeout_seconds = float(sol_cfg.get("timeout_seconds", 30.0))
    max_iterations = int(sol_cfg.get("max_iterations", 100))

    interval_sec = float(sol_cfg.get("interval_minutes", 0)) * 60.0
    websocket_url = (
        sol_cfg.get("helius_ws_url") if sol_cfg.get("use_ws", False) else None
    )
    raydium_program_id = sol_cfg.get("raydium_program_id")
    url = sol_cfg.get("url")
    min_liquidity = float(sol_cfg.get("min_liquidity", 0.0))
    min_tx = int(sol_cfg.get("min_tx_count", 0))

    watcher = PoolWatcher(
        url,
        interval_sec or None,
        websocket_url,
        raydium_program_id,
        min_liquidity=min_liquidity,
    )

    tokens: List[str] = []
    seen: set[str] = set()
    start = asyncio.get_running_loop().time()
    iterations = 0
    try:
        async for event in watcher.watch():
            iterations += 1
            if (
                iterations > max_iterations
                or asyncio.get_running_loop().time() - start > timeout_seconds
            ):
                watcher.stop()
                break
            if event.liquidity < min_liquidity or event.tx_count < min_tx:
                continue
            mint = event.token_mint
            if not mint or mint in seen or mint in TOKEN_MINTS.values():
                continue
            seen.add(mint)
            tokens.append(mint)
            if len(tokens) >= max_tokens:
                watcher.stop()
                break
    finally:
        watcher.stop()
    async def _scan() -> List[str]:
        tokens: List[str] = []
        seen: set[str] = set()
        try:
            async for event in watcher.watch():
                if event.liquidity < min_liquidity or event.tx_count < min_tx:
                    continue
                mint = event.token_mint
                if not mint or mint in seen or mint in TOKEN_MINTS.values():
                    continue
                seen.add(mint)
                tokens.append(mint)
                if len(tokens) >= max_tokens:
                    watcher.stop()
                    break
        except asyncio.CancelledError:
            pass
        finally:
            watcher.stop()
        return tokens

    task = asyncio.create_task(_scan())
    try:
        if timeout_seconds:
            return await asyncio.wait_for(task, timeout_seconds)
        return await task
    except asyncio.TimeoutError:
        logger.warning(
            "Solana token scan timed out after %.1f seconds", timeout_seconds
        )
        task.cancel()
        try:
            return await task
        except asyncio.CancelledError:
            return []

