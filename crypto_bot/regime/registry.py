import io, os, json, pickle, hashlib
from supabase import create_client


def _client():
    url = os.environ["SUPABASE_URL"]
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_SERVICE_KEY")
        or os.environ["SUPABASE_KEY"]
    )
    return create_client(url, key)


def resolve_latest(symbol: str, bucket=None, prefix=None) -> dict:
    bucket = bucket or os.environ.get("CT_MODELS_BUCKET", "models")
    prefix = prefix or os.environ.get("CT_REGIME_PREFIX", "models/regime")
    latest_key = f"{prefix}/{symbol}/LATEST.json"
    sb = _client()
    b = sb.storage.from_(bucket).download(latest_key)
    meta = json.loads(b.decode("utf-8"))
    assert meta.get("key"), "LATEST.json missing 'key'"
    return meta | {"latest_key": latest_key, "bucket": bucket}


def _sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def load_latest_regime(symbol: str):
    try:
        meta = resolve_latest(symbol)
        sb = _client()
        blob = sb.storage.from_(meta["bucket"]).download(meta["key"])
        if "hash" in meta:
            assert _sha256_bytes(blob) == meta["hash"], "Model hash mismatch"
        model = pickle.loads(blob)
        return model, meta
    except Exception as e:
        # Fallback to embedded base64 model
        from crypto_bot.regime import model_data
        model = model_data.load_default()
        return model, {"source": "embedded", "error": str(e)}
from __future__ import annotations

import io
import os
from typing import Tuple, Any

from .ml_fallback import load_model as _load_fallback


def _load_bytes(blob: bytes) -> Any:
    """Deserialize a model from raw bytes.

    Joblib is attempted first and we fall back to pickle. This mirrors the
    logic used elsewhere in the code base and keeps the dependency optional.
    """
    try:
        import joblib  # type: ignore

        return joblib.load(io.BytesIO(blob))
    except Exception:  # pragma: no cover - joblib may be missing or fail
        import pickle

        return pickle.loads(blob)


def load_latest_regime(symbol: str = "BTCUSDT") -> Tuple[Any, dict]:
    """Load the most recent regime model.

    The loader prefers a model stored on Supabase when the required
    credentials are present.  If that retrieval fails for any reason, the
    embedded base64 fallback model is used instead.  The returned tuple
    contains the model object and a metadata dictionary describing the
    source.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    file_name = os.getenv("SUPABASE_MODEL_FILE", "regime_lgbm.pkl")

    if url and key:
        try:  # pragma: no cover - optional dependency and network access
            from supabase import create_client  # type: ignore

            client = create_client(url, key)
            data = client.storage.from_("models").download(file_name)
            model = _load_bytes(data)
            return model, {"source": "supabase", "symbol": symbol}
        except Exception:
            pass

    # Fallback to the embedded model when Supabase is unavailable.
    model = _load_fallback()
    return model, {"source": "fallback", "symbol": symbol}
