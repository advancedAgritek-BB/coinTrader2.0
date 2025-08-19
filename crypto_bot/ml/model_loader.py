import os, json, logging, requests
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

def _supabase_key() -> Optional[str]:
    # Prefer canonical service key, then legacy names
    return (
        os.getenv("SUPABASE_SERVICE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_API_KEY")
    )

def _norm_symbol(symbol: str) -> str:
    # normalize to the naming convention used in storage
    return symbol.replace("/", "_").replace(":", "_")

def load_regime_model(symbol: str) -> Dict[str, Any]:
    bucket = os.getenv("CT_MODELS_BUCKET", "models")
    prefix = os.getenv("CT_REGIME_PREFIX", "regime").strip("/")

    key = f"{prefix}/{_norm_symbol(symbol)}.json"
    url = os.getenv("SUPABASE_URL")
    sb_key = _supabase_key()

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
