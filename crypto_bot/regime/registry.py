from __future__ import annotations

import hashlib
import io
import json
import os
from typing import Any

from .ml_fallback import load_model as _load_fallback


def load_latest_regime(symbol: str) -> Any:
    """Load the most recent regime model for ``symbol``.

    The function attempts to fetch model metadata from Supabase storage.
    The metadata file (``LATEST.json``) is expected to contain a pointer to
    the model binary and optionally its SHA256 hash.  When the download or
    validation fails for any reason, an embedded fallback model is returned
    instead.
    """
    try:  # pragma: no cover - network and optional dependency
        from supabase import create_client  # type: ignore

        url = os.environ["SUPABASE_URL"]
        key = (
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or os.environ.get("SUPABASE_SERVICE_KEY")
            or os.environ["SUPABASE_KEY"]
        )
        bucket = os.environ.get("CT_MODELS_BUCKET", "models")
        prefix = os.environ.get("CT_REGIME_PREFIX", "models/regime")

        client = create_client(url, key)
        latest_key = f"{prefix}/{symbol}/LATEST.json"
        meta_bytes = client.storage.from_(bucket).download(latest_key)
        meta = json.loads(meta_bytes.decode("utf-8"))
        assert meta.get("key"), "LATEST.json missing 'key'"

        blob = client.storage.from_(bucket).download(meta["key"])
        if "hash" in meta:
            digest = "sha256:" + hashlib.sha256(blob).hexdigest()
            assert digest == meta["hash"], "Model hash mismatch"

        try:
            import joblib  # type: ignore

            return joblib.load(io.BytesIO(blob))
        except Exception:  # pragma: no cover - joblib may be missing or fail
            import pickle

            return pickle.loads(blob)
    except Exception:
        return _load_fallback()
