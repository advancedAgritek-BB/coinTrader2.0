import json
from pathlib import Path

from .logger import LOG_DIR, setup_logger

PAIR_FILE = Path(__file__).resolve().parents[2] / "cache" / "liquid_pairs.json"
logger = setup_logger(__name__, LOG_DIR / "pair_cache.log")


def load_liquid_pairs() -> list[str] | None:
    """Return cached list of liquid trading pairs if available."""
    if PAIR_FILE.exists():
        try:
            return json.loads(PAIR_FILE.read_text())
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to read %s: %s", PAIR_FILE, exc)
    return None

