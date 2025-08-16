from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, Optional

from crypto_bot.regime import registry as _registry

logger = logging.getLogger(__name__)

# In-memory caches for loaded models and pointer hashes
_cache: Dict[str, object] = {}
_hashes: Dict[str, str] = {}


def load_latest_regime(symbol: str) -> Optional[object]:
    """Load the latest regime model for ``symbol`` using the registry helpers."""

    try:
        model, _meta = _registry.load_latest_regime(symbol)
        return model
    except Exception:  # pragma: no cover - registry failure
        return None


def maybe_refresh_model(symbol: str) -> None:
    """Reload model if the ``LATEST.json`` metadata has changed."""

    try:
        meta = _registry.resolve_latest(symbol)
    except Exception as exc:  # pragma: no cover - Supabase failure
        logger.debug("Pointer fetch failed for %s: %s", symbol, exc)
        try:
            model, meta = _registry.load_latest_regime(symbol)
            _cache[symbol] = model
            _hashes[symbol] = hashlib.sha256(
                json.dumps(meta, sort_keys=True).encode()
            ).hexdigest()
        except Exception as exc2:  # pragma: no cover - fallback failure
            logger.error("Failed to load regime model for %s: %s", symbol, exc2)
        return

    meta_hash = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    if _hashes.get(symbol) == meta_hash:
        return

    try:
        model, _ = _registry.load_latest_regime(symbol)
        _cache[symbol] = model
        _hashes[symbol] = meta_hash
        logger.info("Regime model refreshed for %s", symbol or "default")
    except Exception as exc:  # pragma: no cover - model load failure
        logger.error("Failed to reload regime model for %s: %s", symbol, exc)
