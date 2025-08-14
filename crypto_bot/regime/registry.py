import io, os, json, pickle, hashlib
from supabase import create_client


def _client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_KEY"]
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
