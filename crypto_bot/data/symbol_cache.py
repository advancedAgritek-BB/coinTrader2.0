import json
import logging
from pathlib import Path
from typing import Dict, Iterable, MutableMapping, MutableSequence, Any

DENY = {"AIBTC/USD", "AIBTC:USD"}
_LOGGED: set[str] = set()

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).resolve().parents[2] / "cache" / "symbol_cache.json"

def _maybe_purge(symbol: str) -> bool:
    if symbol in DENY:
        if symbol not in _LOGGED:
            logger.info("Purged denylisted symbol from cache: %s", symbol)
            _LOGGED.add(symbol)
        return False
    return True

def purge_denylisted(container: Any) -> Any:
    """Remove denylisted symbols from ``container`` in-place and return it.

    ``container`` may be a list of symbols or a mapping of symbol -> value.
    """
    if isinstance(container, MutableSequence):
        items = [s for s in container if _maybe_purge(str(s))]
        container[:] = items
    elif isinstance(container, MutableMapping):
        for key in list(container.keys()):
            if not _maybe_purge(str(key)):
                container.pop(key, None)
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
