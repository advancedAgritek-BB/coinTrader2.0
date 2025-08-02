"""Token account helpers for Solana.

This module fetches token accounts for a wallet, enriches them with metadata
from the Helius API and scores them using a machine learning model. Only
accounts whose model probability exceeds a configurable threshold are returned.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import aiohttp

logger = logging.getLogger(__name__)

MIN_BALANCE_THRESHOLD = float(os.getenv("MIN_BALANCE_THRESHOLD", "0.0"))
ML_SCORE_THRESHOLD = float(os.getenv("ML_SCORE_THRESHOLD", "0.5"))


async def enrich_with_metadata(
    account_pubkey: str, session: aiohttp.ClientSession
) -> Dict[str, Any]:
    """Return metadata for ``account_pubkey`` using the Helius assets endpoint."""

    api_key = os.getenv("HELIUS_KEY")
    if not api_key:
        raise ValueError("HELIUS_KEY environment variable not set")

    url = f"https://api.helius.xyz/v0/assets/{account_pubkey}?api-key={api_key}"
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return {}
            return await resp.json()
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Metadata fetch failed: %s", exc)
        return {}


def predict_token_regime(token_data: Dict[str, Any]) -> float:
    """Return the maximum regime probability for ``token_data``.

    The function loads the ``regime_lgbm`` model via ``load_model`` and predicts
    using basic features such as liquidity and transaction count. On any
    failure the error is logged and ``0.0`` is returned.
    """

    try:  # pragma: no cover - optional dependency
        from coinTrader_Trainer.ml_trainer import load_model

        model = load_model("regime_lgbm")
        features = [
            float(token_data.get("liquidity", 0)),
            float(token_data.get("tx_count", token_data.get("transaction_count", 0))),
        ]
        try:
            pred = (
                model.predict_proba([features])[0]
                if hasattr(model, "predict_proba")
                else model.predict([features])[0]
            )
        except Exception:  # pragma: no cover - best effort
            pred = model.predict([features])[0]
        return float(max(pred)) if hasattr(pred, "__iter__") else float(pred)
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("Token regime prediction failed: %s", exc)
        return 0.0


async def get_token_accounts(
    wallet_address: str,
    threshold: float | None = None,
    ml_threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Return SPL token accounts above ``threshold`` with ML scores.

    Accounts are enriched with metadata and scored via
    :func:`predict_token_regime`. Only accounts whose score exceeds
    ``ml_threshold`` are included in the result.
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
    ml_threshold = float(
        ml_threshold if ml_threshold is not None else ML_SCORE_THRESHOLD
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()

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

                if float(amount) < threshold:
                    continue

                meta = await enrich_with_metadata(info.get("mint", ""), session)
                ml_score = predict_token_regime(meta)
                acc["metadata"] = meta
                acc["ml_score"] = ml_score
                if ml_score >= ml_threshold:
                    filtered.append(acc)

            return filtered
    except aiohttp.ClientError as exc:  # pragma: no cover - network
        logger.error("Helius request failed: %s", exc)
        raise


async def get_token_accounts_ml_filter(
    wallet_address: str,
    threshold: float | None = None,
    ml_threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Return enriched token accounts passing an ML probability filter.

    Wrapper around :func:`get_token_accounts` kept for backwards compatibility.
    """

    return await get_token_accounts(wallet_address, threshold, ml_threshold)

