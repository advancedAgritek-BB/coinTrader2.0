from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, Optional

try:  # registry may be unavailable in some environments
    from cointrainer import registry as _registry  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _registry = None

from .api import _load_model_from_bytes

logger = logging.getLogger(__name__)

# In-memory caches for loaded models and pointer hashes
_cache: Dict[str, object] = {}
_hashes: Dict[str, str] = {}


def load_latest_regime(symbol: str) -> Optional[object]:
    """Load the latest regime model for ``symbol`` from the registry."""
    if _registry is None:  # pragma: no cover - registry optional
        return None
    prefix = f"models/regime/{symbol}/" if symbol else "models/regime/"
    blob = _registry.load_latest(prefix, allow_fallback=False)  # type: ignore[attr-defined]
    return _load_model_from_bytes(blob)


def maybe_refresh_model(symbol: str) -> None:
    """Reload model if the registry pointer has changed."""
    if _registry is None:  # pragma: no cover - registry optional
        return
    prefix = f"models/regime/{symbol}/" if symbol else "models/regime/"
    try:
        meta = _registry.load_pointer(prefix)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - registry failure
        logger.debug("Pointer fetch failed for %s: %s", symbol, exc)
        return
    meta_hash = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    if _hashes.get(symbol) == meta_hash:
        return
    _hashes[symbol] = meta_hash
    try:
        model = load_latest_regime(symbol)
        if model is not None:
            _cache[symbol] = model
            logger.info("Regime model refreshed for %s", symbol or "default")
    except Exception as exc:  # pragma: no cover - model load failure
        logger.error("Failed to reload regime model for %s: %s", symbol, exc)
