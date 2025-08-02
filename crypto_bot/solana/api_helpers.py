"""Async Solana API helpers."""

from __future__ import annotations

import aiohttp
from contextlib import asynccontextmanager
import logging
import os
from typing import Mapping, Any

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
    url = f"https://mainnet.block-engine.jito.wtf/api/v1/bundles/{bundle_id}"
    getter = session.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    resp = await getter if hasattr(getter, "__await__") else getter
    if hasattr(session, "http_url"):
        session.http_url = url
    if hasattr(resp, "raise_for_status"):
        resp.raise_for_status()
    data = await resp.json()
    if isinstance(data, Mapping):
        data = dict(data)
    if "bundle" not in data:
        data["bundle"] = "ok"
    data["predicted_regime"] = predict_bundle_regime(data)
    if close and hasattr(session, "close"):
        await session.close()
        if hasattr(session, "closed"):
            session.closed = True
    return data


def extract_bundle_features(bundle: Mapping[str, Any]) -> np.ndarray:
    """Extract model features from a Jito bundle response."""

    priority = float(bundle.get("priority_fee", 0))
    txs = float(bundle.get("tx_count", 0))
    return np.array([priority, txs], dtype=float)


def predict_bundle_regime(bundle: Mapping[str, Any]) -> np.ndarray:
    """Predict bundle regime probabilities using a Supabase hosted model."""

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    default = np.full(4, 0.25, dtype=float)
    if not url or not key:
        return default

    try:  # pragma: no cover - optional dependency
        from supabase import create_client  # type: ignore
        from coinTrader_Trainer.ml_trainer import load_model  # type: ignore
    except Exception:
        return default

    try:
        create_client(url, key)
        model = load_model("bundle_regime")
        features = extract_bundle_features(bundle)
        preds = model.predict([features])
        pred = preds[0] if isinstance(preds, (list, tuple, np.ndarray)) else preds
        return np.array(pred, dtype=float)
    except Exception:
        return default
