"""Utilities for loading regime models.

This module centralizes retrieval of ML models used by the
:mod:`crypto_bot.regime` package. Models are fetched from Supabase when
credentials are available. When the download fails the loader falls back to an
HTTP URL and finally to a local file path.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from io import BytesIO
from typing import Tuple

logger = logging.getLogger(__name__)


def _norm_symbol(sym: str) -> str:
    """Return ``sym`` in a canonical form.

    Examples
    --------
    >>> _norm_symbol("btc/usdt")
    'BTC_USDT'
    """

    sym = sym.upper().replace("XBT", "BTC")
    sym = sym.replace("-", "_").replace("/", "_")
    return sym


def _supabase_key(symbol: str, *parts: str) -> str:
    """Construct a storage key for ``symbol`` within the regime prefix."""

    prefix = os.getenv("CT_REGIME_PREFIX", "models/regime")
    folder = _norm_symbol(symbol).replace("_", "")
    return "/".join([prefix, folder, *parts])


def _deserialize(blob: bytes) -> Tuple[object | None, object | None]:
    """Deserialize a model blob returning ``(model, scaler)``."""

    try:
        obj = pickle.loads(blob)
    except Exception:
        try:  # pragma: no cover - optional dependency
            import joblib

            obj = joblib.load(BytesIO(blob))
        except Exception:
            return None, None

    if isinstance(obj, dict):
        return obj.get("model"), obj.get("scaler")
    return obj, None


async def load_regime_model(symbol: str) -> Tuple[object | None, object | None, str | None]:
    """Load the regime model for ``symbol``.

    The function tries multiple sources in order:

    1. Supabase storage using ``SUPABASE_URL``/``SUPABASE_SERVICE_ROLE_KEY``
       (or compatible keys). The model is looked up under
       ``CT_MODELS_BUCKET`` and ``CT_REGIME_PREFIX`` using ``LATEST.json``.
    2. HTTP URL specified by ``CT_MODEL_FALLBACK_URL`` or the corresponding
       configuration value.
    3. Local file path specified by ``CT_MODEL_LOCAL_PATH``.

    The return value is ``(model, scaler, path)`` where ``path`` describes the
    source used. ``model`` and ``scaler`` are ``None`` when loading fails.
    """

    bucket = os.getenv("CT_MODELS_BUCKET", "models")
    template = os.getenv("CT_REGIME_MODEL_TEMPLATE", "{symbol_lower}_regime_lgbm.pkl")

    # Supabase -------------------------------------------------------------
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_API_KEY")
    )
    if url and key:
        try:  # pragma: no cover - optional dependency
            from supabase import create_client  # type: ignore

            client = create_client(url, key)
            latest_key = _supabase_key(symbol, "LATEST.json")
            model_bytes = None
            try:
                meta_bytes = client.storage.from_(bucket).download(latest_key)
                meta = json.loads(meta_bytes.decode("utf-8"))
                model_key = meta.get("key")
                if model_key:
                    model_bytes = client.storage.from_(bucket).download(model_key)
                    model_path = model_key
            except Exception:
                model_key = _supabase_key(
                    symbol,
                    template.format(
                        symbol=_norm_symbol(symbol).replace("_", ""),
                        symbol_lower=_norm_symbol(symbol).replace("_", "").lower(),
                    ),
                )
                try:
                    model_bytes = client.storage.from_(bucket).download(model_key)
                    model_path = model_key
                except Exception:
                    model_bytes = None
            if model_bytes:
                model, scaler = _deserialize(model_bytes)
                if model is not None:
                    logger.info("Loaded regime model from Supabase: %s", model_path)
                    return model, scaler, model_path
        except Exception as exc:
            logger.warning("Supabase download failed: %s", exc)

    # HTTP fallback --------------------------------------------------------
    fallback_url = os.getenv("CT_MODEL_FALLBACK_URL")
    if fallback_url:
        try:
            import urllib.request

            with urllib.request.urlopen(fallback_url) as resp:  # nosec: B310
                blob = resp.read()
            model, scaler = _deserialize(blob)
            if model is not None:
                logger.info(
                    "Loaded fallback regime model from %s", fallback_url
                )
                return model, scaler, fallback_url
        except Exception as exc:
            logger.warning("HTTP fallback failed: %s", exc)

    # Local file fallback --------------------------------------------------
    local_path = os.getenv("CT_MODEL_LOCAL_PATH")
    if local_path:
        try:
            with open(local_path, "rb") as f:
                blob = f.read()
            model, scaler = _deserialize(blob)
            if model is not None:
                logger.info(
                    "Loaded local fallback regime model from %s", local_path
                )
                return model, scaler, local_path
        except Exception as exc:
            logger.warning("Local model load failed: %s", exc)

    return None, None, None

import os, json, logging, requests
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def supabase_key() -> Optional[str]:
    """Return the Supabase key from environment variables."""
    return os.getenv("SUPABASE_KEY")

def _norm_symbol(symbol: str) -> str:
    # normalize to the naming convention used in storage
    return symbol.replace("/", "_").replace(":", "_")

def load_regime_model(symbol: str) -> Dict[str, Any]:
    bucket = os.getenv("CT_MODELS_BUCKET", "models")
    prefix = os.getenv("CT_REGIME_PREFIX", "regime").strip("/")

    key = f"{prefix}/{_norm_symbol(symbol)}.json"
    url = os.getenv("SUPABASE_URL")
    sb_key = supabase_key()

    # 1) Supabase storage client path (if you use supabase-py)
    try:
        from supabase import create_client  # type: ignore
        if url and sb_key:
            supa = create_client(url, sb_key)
            data = supa.storage.from_(bucket).download(key)
            log.info("Loaded regime model from Supabase: %s/%s", bucket, key)
            return json.loads(data)
    except Exception as e:
        log.warning("Supabase storage download failed (%s); trying public URL fallback", e)

    # 2) Public HTTP fallback (works if the bucket/object is public)
    try:
        if url:
            http = f"{url.rstrip('/')}/storage/v1/object/public/{bucket}/{key}"
            r = requests.get(http, timeout=10)
            if r.ok:
                log.info("Loaded regime model via public URL: %s", http)
                return r.json()
            else:
                log.warning("Public URL returned %s for %s", r.status_code, http)
    except Exception as e:
        log.warning("Public URL fetch failed: %s", e)

    # 3) Local file fallback
    local = Path("crypto_bot") / "models" / "regime" / f"{_norm_symbol(symbol)}.json"
    if local.exists():
        log.info("Loaded local regime model: %s", local)
        return json.loads(local.read_text())

    raise FileNotFoundError(f"Regime model not found for {symbol} in Supabase or local path")
