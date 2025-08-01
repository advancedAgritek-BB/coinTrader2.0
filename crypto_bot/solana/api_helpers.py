"""Async Solana API helpers."""

from __future__ import annotations

import aiohttp
from contextlib import asynccontextmanager
import logging
import os
from typing import Mapping, Any

import numpy as np
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
    ws = await session.ws_connect(url, timeout=30)
    try:
        try:
            yield ws
        except Exception as exc:
            logger.error("Error while using Helius websocket", exc_info=exc)
            raise
    finally:
        await ws.close()
        await session.close()


async def fetch_jito_bundle(bundle_id: str, api_key: str, session: aiohttp.ClientSession | None = None):
    """Fetch a bundle status from Jito Block Engine."""

    close = False
    if session is None:
        session = aiohttp.ClientSession()
        close = True
    async with session.get(
        f"https://mainnet.block-engine.jito.wtf/api/v1/bundles/{bundle_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    if close:
        await session.close()
    if isinstance(data, Mapping):
        data = dict(data)
        data["predicted_regime"] = predict_bundle_regime(data)
    return data


def predict_bundle_regime(bundle: Mapping[str, Any]) -> str:
    """Predict bundle regime using a Supabase hosted model."""

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return "unknown"

    try:  # pragma: no cover - optional dependency
        from supabase import create_client  # type: ignore
        from coinTrader_Trainer.ml_trainer import load_model  # type: ignore
    except Exception:
        return "unknown"

    try:
        create_client(url, key)
        model = load_model("bundle_regime")
        priority = float(bundle.get("priority_fee", 0))
        txs = float(bundle.get("tx_count", 0))
        preds = model.predict([[priority, txs]])
        pred = preds[0] if isinstance(preds, (list, tuple, np.ndarray)) else preds
        if isinstance(pred, (list, tuple, np.ndarray)):
            idx = int(np.argmax(pred))
            labels = ["stable", "volatile"]
            return labels[idx] if idx < len(labels) else str(idx)
        return str(pred)
    except Exception:
        return "unknown"

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
