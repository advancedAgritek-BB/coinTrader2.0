"""Utilities for loading ML regime models from Supabase or local files."""

from __future__ import annotations

import logging
import os
import pickle
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import requests

log = logging.getLogger(__name__)



def _supabase_key() -> Optional[str]:
    """Return the Supabase key from environment variables."""
    return os.getenv("SUPABASE_KEY")


def _norm_symbol(symbol: str) -> str:
    """Normalise exchange symbols to storage naming convention."""

    return symbol.replace("/", "_").replace(":", "_").upper()


def _deserialize(data: bytes) -> Tuple[object | None, object | None]:
    """Deserialize model data into (model, scaler)."""

    try:
        obj = pickle.loads(data)
    except Exception:
        try:  # pragma: no cover - optional dependency
            import joblib

            obj = joblib.load(BytesIO(data))
        except Exception as exc:  # pragma: no cover - joblib optional
            log.error("Failed to deserialize regime model: %s", exc)
            return None, None

    if isinstance(obj, dict):
        return obj.get("model"), obj.get("scaler")
    return obj, None


def load_regime_model(symbol: str) -> Tuple[object | None, object | None, str | None]:
    """Load a regime model for ``symbol`` with multiple fallbacks.

    The loader attempts three strategies in order:

    1. Supabase storage using ``supabase-py`` if available.
    2. Direct HTTP request to the public object URL.
    3. Local file located under ``crypto_bot/models/regime``.

    ``CT_MODELS_BUCKET`` specifies the bucket name (default ``"models"``) and
    ``CT_REGIME_PREFIX`` controls the prefix/path within the bucket (default
    ``"regime"``).
    """

    bucket = os.getenv("CT_MODELS_BUCKET", "models")
    prefix = os.getenv("CT_REGIME_PREFIX", "regime").strip("/")
    norm = _norm_symbol(symbol)
    filename = f"{norm.lower()}_regime_lgbm.pkl"
    key = f"{prefix}/{filename}" if prefix else filename

    url = os.getenv("SUPABASE_URL")
    sb_key = _supabase_key()

    # 1) Supabase storage client path
    if url and sb_key:
        try:  # pragma: no cover - supabase optional
            from supabase import create_client  # type: ignore

            supa = create_client(url, sb_key)
            data = supa.storage.from_(bucket).download(key)
            model, scaler = _deserialize(data)
            if model is not None:
                log.info("Loaded regime model from Supabase: %s/%s", bucket, key)
                return model, scaler, key
        except Exception as exc:
            log.warning(
                "Supabase storage download failed (%s); trying public URL fallback",
                exc,
            )

    # 2) Public HTTP fallback
    if url:
        http_url = f"{url.rstrip('/')}/storage/v1/object/public/{bucket}/{key}"
        try:
            resp = requests.get(http_url, timeout=10)
            if resp.ok:
                model, scaler = _deserialize(resp.content)
                if model is not None:
                    log.info("Loaded regime model via public URL: %s", http_url)
                    return model, scaler, http_url
            else:
                log.warning("Public URL returned %s for %s", resp.status_code, http_url)
        except Exception as exc:
            log.warning("Public URL fetch failed: %s", exc)

    # 3) Local file fallback
    local_path = Path("crypto_bot") / "models" / "regime" / filename
    if local_path.exists():
        try:
            data = local_path.read_bytes()
            model, scaler = _deserialize(data)
            if model is not None:
                log.info("Loaded local regime model: %s", local_path)
                return model, scaler, str(local_path)
        except Exception as exc:
            log.warning("Local regime model load failed: %s", exc)

    return None, None, None

