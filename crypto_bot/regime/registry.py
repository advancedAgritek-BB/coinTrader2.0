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
