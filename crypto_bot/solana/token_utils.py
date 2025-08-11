from __future__ import annotations

import os
import logging
from typing import List, Dict, Any

import aiohttp

logger = logging.getLogger(__name__)

MIN_BALANCE_THRESHOLD = float(os.getenv("MIN_BALANCE_THRESHOLD", "0.0"))


async def get_token_accounts(wallet_address: str, threshold: float | None = None) -> List[Dict[str, Any]]:
    """Return SPL token accounts with balances above ``threshold``.

    Parameters
    ----------
    wallet_address: str
        The Solana wallet address to query.
    threshold: float, optional
        Minimum balance required. Defaults to ``MIN_BALANCE_THRESHOLD``.
    """

    api_key = os.getenv("HELIUS_KEY")
    if not api_key:
        raise ValueError("HELIUS_KEY environment variable not set")

    url = f"https://mainnet.helius-rpc.com/v1/?api-key={api_key}"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            wallet_address,
            {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
            {"encoding": "jsonParsed"},
        ],
    }

    threshold = float(threshold if threshold is not None else MIN_BALANCE_THRESHOLD)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except aiohttp.ClientError as exc:  # pragma: no cover - network
        logger.error("Helius request failed: %s", exc)
        raise

    accounts = data.get("result", {}).get("value", [])
    filtered: List[Dict[str, Any]] = []
    for acc in accounts:
        info = (
            acc.get("account", {})
            .get("data", {})
            .get("parsed", {})
            .get("info", {})
        )
        amount_info = info.get("tokenAmount", {})
        amount = amount_info.get("uiAmount")
        if amount is None:
            try:
                amount = float(amount_info.get("uiAmountString", 0))
            except (ValueError, TypeError):
                amount = 0.0
        if float(amount) >= threshold:
            filtered.append(acc)

    return filtered
