"""Regime model loading utilities.

This module provides a small helper to fetch the latest regime model from
Supabase storage.  The function is resilient: if the Supabase dependency is not
available or the expected files cannot be retrieved the code gracefully falls
back to a built-in heuristic model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Tuple

from .ml_fallback import load_model as _load_fallback


logger = logging.getLogger(__name__)
_no_model_logged = False


def load_latest_regime(symbol: str) -> Tuple[Any, Dict]:
    """Load the most recent regime model for ``symbol``.

    The function first attempts to download ``LATEST.json`` from the Supabase
    bucket which points to the actual model file and optionally contains a
    SHA256 checksum.  When the metadata or model file is missing the code tries a
    direct path and finally falls back to the embedded heuristic model.
    """

    bucket = os.environ.get("CT_MODELS_BUCKET", "models")
    prefix = os.environ.get("CT_REGIME_PREFIX", "regime")
    template = os.environ.get(
        "CT_REGIME_MODEL_TEMPLATE",
        "{prefix}/{symbol}/{symbol_lower}_regime_lgbm.pkl",
    )

    client = None
    try:  # pragma: no cover - optional dependency and network access
        from supabase import create_client  # type: ignore

        url = os.environ["SUPABASE_URL"]
        key = (
            os.environ.get("SUPABASE_SERVICE_KEY")
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or os.environ.get("SUPABASE_KEY")
        )
        client = create_client(url, key)

        latest_key = f"{prefix}/{symbol}/LATEST.json"
        meta_bytes = client.storage.from_(bucket).download(latest_key)
        meta = json.loads(meta_bytes.decode("utf-8"))
        assert meta.get("key"), "LATEST.json missing 'key'"

        blob = client.storage.from_(bucket).download(meta["key"])
        if "hash" in meta:
            digest = "sha256:" + hashlib.sha256(blob).hexdigest()
            assert digest == meta["hash"], "Model hash mismatch"
        return blob, meta
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        msg = str(getattr(exc, "message", exc))
        not_found = status == 404 or "404" in str(exc) or "not_found" in msg

        if client and not_found:
            for direct_key in (
                f"{prefix}/{symbol}/{symbol.lower()}_regime_lgbm.pkl",
                template.format(
                    prefix=prefix, symbol=symbol, symbol_lower=symbol.lower()
                ),
            ):
                try:
                    blob = client.storage.from_(bucket).download(direct_key)
                    return blob, {}
                except Exception as exc2:  # pragma: no cover - network
                    status2 = getattr(
                        getattr(exc2, "response", None), "status_code", None
                    )
                    if status2 != 404 and "404" not in str(exc2):
                        raise

        if not_found or client is None:
            # Attempt HTTP fallback when Supabase download is unavailable
            fallback_url = os.getenv("CT_MODEL_FALLBACK_URL")
            if not fallback_url:
                try:
                    from crypto_bot import main as _main  # type: ignore

                    cfg = getattr(_main, "_LAST_ML_CFG", {}) or {}
                    if isinstance(cfg, dict):
                        fallback_url = cfg.get("model_fallback_url")
                except Exception:  # pragma: no cover - circular import or missing cfg
                    fallback_url = None

            if fallback_url:
                try:
                    import urllib.request

                    with urllib.request.urlopen(fallback_url) as resp:
                        blob = resp.read()
                    return blob, {"source": fallback_url}
                except Exception as url_exc:
                    logger.info(
                        "Failed to download fallback model from %s: %s",
                        fallback_url,
                        url_exc,
                    )

            global _no_model_logged
            if not _no_model_logged:
                logger.info(
                    "No regime model found in bucket '%s/%s' — falling back to heuristics (this is OK for live trading)",
                    bucket,
                    prefix,
                )
                _no_model_logged = True
            return _load_fallback(), {}
        raise

