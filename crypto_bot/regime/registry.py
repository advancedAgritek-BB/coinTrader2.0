from __future__ import annotations

import hashlib
import json
import os
import logging
from typing import Any, Tuple, Dict

from .ml_fallback import load_model as _load_fallback


logger = logging.getLogger(__name__)
_no_model_logged = False


def load_latest_regime(symbol: str) -> Tuple[Any, Dict]:
    """Load the most recent regime model for ``symbol``.

    The function attempts to fetch model metadata from Supabase storage.  The
    metadata file (``LATEST.json``) is expected to contain a pointer to the
    model binary and optionally its SHA256 hash.  When the metadata file is not
    found the function looks for a direct model file named
    ``<symbol_lower>_regime_lgbm.pkl`` (customisable via the
    ``CT_REGIME_MODEL_TEMPLATE`` environment variable).  If neither the
    metadata nor the direct file can be retrieved the embedded fallback model
    is returned and a single info message is logged.  Other failures are raised
    so callers can handle them separately.
    """

    bucket = os.environ.get("CT_MODELS_BUCKET", "models")
    prefix = os.environ.get("CT_REGIME_PREFIX", "models/regime")
    template = os.environ.get(
        "CT_REGIME_MODEL_TEMPLATE",
        "{prefix}/{symbol}/{symbol_lower}_regime_lgbm.pkl",
    )

    try:  # pragma: no cover - network and optional dependency
        from supabase import create_client  # type: ignore

        url = os.environ["SUPABASE_URL"]
        key = (
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or os.environ.get("SUPABASE_SERVICE_KEY")
            or os.environ["SUPABASE_KEY"]
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
        if status == 404 or "404" in str(exc):
            try:
                direct_key = template.format(
                    prefix=prefix, symbol=symbol, symbol_lower=symbol.lower()
                )
                blob = client.storage.from_(bucket).download(direct_key)
                return blob, {}
            except Exception as exc2:  # pragma: no cover - network
                status2 = getattr(getattr(exc2, "response", None), "status_code", None)
                if status2 != 404 and "404" not in str(exc2):
                    raise
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
