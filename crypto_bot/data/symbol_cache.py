import json
import logging
from pathlib import Path
from typing import Any, Dict, MutableMapping, MutableSequence

logger = logging.getLogger(__name__)

# The cache file lives under ``cache/symbol_cache.json`` relative to the
# repository root.
CACHE_FILE = Path(__file__).resolve().parents[2] / "cache" / "symbol_cache.json"

try:  # pragma: no cover - best effort; cfg may not be available in tests
    from crypto_bot.config import cfg  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    from types import SimpleNamespace

    cfg = SimpleNamespace(denylist_symbols=[])  # type: ignore


def purge_denylisted(container: Any) -> Any:
    """Remove denylisted symbols from ``container`` in-place and return it.

    ``container`` may be a list of symbols or a mapping of ``symbol -> value``.
    The denylist is sourced from ``cfg.denylist_symbols`` allowing runtime
    configuration. Any purged symbol is logged for observability.
    """

    deny = set(getattr(cfg, "denylist_symbols", []) or [])
    if not deny:
        return container
    if isinstance(container, MutableSequence):
        restored = [s for s in container if s not in deny]
        for s in set(container).intersection(deny):
            logger.info("Purged denylisted symbol from cache: %s", s)
        container[:] = restored
    elif isinstance(container, MutableMapping):
        for key in list(container.keys()):
            if key in deny:
                container.pop(key, None)
                logger.info("Purged denylisted symbol from cache: %s", key)
    return container

def load_cache() -> Dict[str, float]:
    """Return cached symbol mapping with denylisted entries removed."""
    if not CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text())
        if isinstance(data, list):
            data = {p: 0.0 for p in data}
        elif not isinstance(data, dict):
            data = {}
        purge_denylisted(data)
        return {str(k): float(v) for k, v in data.items()}
    except Exception:  # pragma: no cover - best effort
        logger.warning("Failed to read %s", CACHE_FILE, exc_info=True)
        return {}

def save_cache(data: Dict[str, float]) -> None:
    """Save ``data`` excluding denylisted symbols."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    purge_denylisted(data)
    with CACHE_FILE.open("w") as f:
        json.dump(data, f)
