"""Utilities for loading ML regime models from Supabase or local files."""

from __future__ import annotations

import logging
import os
import pickle
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import urllib.request

log = logging.getLogger(__name__)



def _supabase_key() -> Optional[str]:
    """Return the Supabase key from environment variables."""
    return os.getenv("SUPABASE_KEY")


def _norm_symbol(symbol: str) -> str:
    """Normalise exchange symbols to storage naming convention."""

    return symbol.replace("/", "_").replace(":", "_").upper()


def _fallback_url(symbol: str) -> Optional[str]:
    """Return configured fallback URL for ``symbol`` if available."""

    tmpl = os.getenv("CT_MODEL_FALLBACK_URL")
    if not tmpl:
        try:
            from crypto_bot import main as _main  # type: ignore

            cfg = getattr(_main, "_LAST_ML_CFG", {}) or {}
            tmpl = cfg.get("model_fallback_url") if isinstance(cfg, dict) else None
        except Exception:  # pragma: no cover - circular import or missing cfg
            tmpl = None
    if tmpl:
        norm = _norm_symbol(symbol)
        return tmpl.format(symbol=norm, symbol_lower=norm.lower())
    return None


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

    The loader first tries Supabase storage. If the object does not exist or
    an error occurs a configurable remote URL is tried as a secondary source.
    When both strategies fail ``None`` is returned and callers should treat
    this as a neutral regime.

    ``CT_MODELS_BUCKET`` specifies the bucket name (default ``"models"``) and
    ``CT_REGIME_PREFIX`` controls the prefix/path within the bucket (default is
    empty, meaning the bucket root).
    """

    bucket = os.getenv("CT_MODELS_BUCKET", "models")
    prefix = os.getenv("CT_REGIME_PREFIX", "").strip("/")
    norm = _norm_symbol(symbol)
    filename = f"{norm.lower()}_regime_lgbm.pkl"
    key = f"{prefix}/{filename}" if prefix else filename

    url = os.getenv("SUPABASE_URL")
    sb_key = _supabase_key()

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
                "Supabase regime model for %s not found (%s); falling back to remote URL",
                symbol,
                exc,
            )

    fb_url = _fallback_url(norm)
    if fb_url:
        try:
            with urllib.request.urlopen(fb_url, timeout=5) as resp:
                data = resp.read()
            model, scaler = _deserialize(data)
            if model is not None:
                log.info("Loaded regime model from fallback URL: %s", fb_url)
                return model, scaler, fb_url
        except Exception as exc:
            log.error(
                "Fallback URL for %s also failed (%s); regime=neutral",
                symbol,
                exc,
            )

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

