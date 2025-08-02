import os
import logging
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
    """Return breakout probability from ``token_data`` via ``regime_lgbm``."""

    try:  # pragma: no cover - optional dependency
        from coinTrader_Trainer.ml_trainer import load_model

        model = load_model("regime_lgbm")
        features = [
            float(token_data.get("liquidity", 0)),
            float(token_data.get("volume", 0)),
            float(token_data.get("tx_count", 0)),
        ]
        pred = model.predict([features])[0]
        return float(pred[1] if isinstance(pred, (list, tuple)) else pred)
    except Exception as exc:
        logger.error("Token regime prediction failed: %s", exc)
        return 0.0


async def get_token_accounts(
    wallet_address: str, threshold: float | None = None
) -> List[Dict[str, Any]]:
    """Return SPL token accounts with balances above ``threshold``.

    Accounts are enriched with metadata and scored via
    :func:`predict_token_regime`. Only accounts with scores above
    ``ML_SCORE_THRESHOLD`` are returned.
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
                    meta = await enrich_with_metadata(info.get("mint", ""), session)
                    ml_score = predict_token_regime(meta)
                    acc["metadata"] = meta
                    acc["ml_score"] = ml_score
                    if ml_score >= ML_SCORE_THRESHOLD:
                        filtered.append(acc)
            return filtered
    except aiohttp.ClientError as exc:  # pragma: no cover - network
        logger.error("Helius request failed: %s", exc)
        raise

