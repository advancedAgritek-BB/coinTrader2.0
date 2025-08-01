"""Async Solana API helpers."""

from __future__ import annotations

import aiohttp
from contextlib import asynccontextmanager
import logging
import os
import numpy as np


logger = logging.getLogger(__name__)

# Base endpoints from the blueprint
# Helius WebSocket: wss://mainnet.helius-rpc.com
# Jito Block Engine REST: https://mainnet.block-engine.jito.wtf/api/v1
# The blueprint recommends staying below 100 requests/sec for Helius and 60 requests/sec for Jito.
# API keys are provided via environment variables HELIUS_KEY and JITO_KEY.


@asynccontextmanager
async def helius_ws(api_key: str):
    """Yield a websocket connection to Helius RPC and close the session."""

    url = f"wss://mainnet.helius-rpc.com/?api-key={api_key}"
    session = aiohttp.ClientSession()
    ws = await session.ws_connect(url)
    try:
        yield ws
    finally:
        await ws.close()
        await session.close()


async def fetch_jito_bundle(bundle_id: str, api_key: str, session: aiohttp.ClientSession | None = None):
    """Fetch a bundle status from Jito Block Engine."""

    close = False
    if session is None:
        session = aiohttp.ClientSession()
        close = True
    try:
        async with session.get(
            f"https://mainnet.block-engine.jito.wtf/api/v1/bundles/{bundle_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        data["predicted_regime"] = predict_bundle_regime(data)
        return data
    except Exception as exc:  # pragma: no cover - network failures
        logger.error("Failed fetching bundle %s: %s", bundle_id, exc)
        return {}
    finally:
        if close:
            await session.close()


def extract_bundle_features(bundle: dict) -> np.ndarray:
    """Return numerical features from a Jito bundle."""

    txs = bundle.get("transactions", [])
    num_txs = len(txs)
    compute_units = 0
    for tx in txs:
        compute_units += int(tx.get("compute_units", tx.get("computeUnits", 0)))
    return np.array([num_txs, compute_units], dtype=float)


def predict_bundle_regime(bundle: dict) -> str:
    """Predict trading regime for a Jito bundle using ML."""

    try:  # pragma: no cover - optional dependency
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise ValueError("Missing Supabase credentials")
        from coinTrader_Trainer.ml_trainer import load_model

        model = load_model("regime_lgbm")
        feats = extract_bundle_features(bundle).reshape(1, -1)
        pred = model.predict(feats)
        return str(pred[0])
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("Bundle regime prediction failed: %s", exc)
        return "unknown"
