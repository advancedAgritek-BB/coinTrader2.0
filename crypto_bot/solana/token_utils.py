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

    url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
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


async def enrich_with_metadata(account: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder that can enrich an account with metadata.

    This function is intentionally simple so that tests can patch it with
    custom logic without requiring network access.
    """

    return account


def predict_token_regime(account: Dict[str, Any]) -> float:
    """Predict trading regime probability for a token account.

    The default implementation returns ``0.0`` and is expected to be patched in
    tests with a model that returns a value between 0 and 1.
    """

    return 0.0


async def get_token_accounts_ml_filter(
    wallet_address: str, threshold: float | None = None
) -> List[Dict[str, Any]]:
    """Return enriched token accounts passing an ML probability filter.

    Accounts are first fetched using :func:`get_token_accounts`, then enriched
    with metadata and scored via :func:`predict_token_regime`. Only accounts with
    a prediction score of at least ``0.5`` are returned.
    """

    accounts = await get_token_accounts(wallet_address, threshold)
    final: List[Dict[str, Any]] = []
    for acc in accounts:
        try:
            enriched = await enrich_with_metadata(acc)
        except Exception:  # pragma: no cover - best effort
            logger.error("metadata enrichment failed", exc_info=True)
            continue

        try:
            score = float(predict_token_regime(enriched))
        except Exception:  # pragma: no cover - best effort
            logger.error("ML prediction failed", exc_info=True)
            score = 0.0

        if score >= 0.5:
            enriched["ml_score"] = score
            final.append(enriched)

    return final
