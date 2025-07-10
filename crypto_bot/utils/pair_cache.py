import json
from pathlib import Path

from .logger import LOG_DIR, setup_logger

PAIR_FILE = Path(__file__).resolve().parents[2] / "cache" / "liquid_pairs.json"
logger = setup_logger(__name__, LOG_DIR / "pair_cache.log")


def load_liquid_map() -> dict[str, float] | None:
    """Return cached mapping of pair -> timestamp if available."""
    if PAIR_FILE.exists():
        try:
            data = json.loads(PAIR_FILE.read_text())
            if isinstance(data, list):
                return {p: 0.0 for p in data}
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to read %s: %s", PAIR_FILE, exc)
    return None


def load_liquid_pairs() -> list[str] | None:
    """Return cached list of liquid trading pairs if available."""
    data = load_liquid_map()
    return list(data) if data else None

