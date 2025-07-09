import json
from pathlib import Path
from typing import Any, Dict

from .logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "bot.log")

STATS_FILE = LOG_DIR / "strategy_stats.json"

# Track whether we've warned about a missing stats file to avoid spam
_missing_warning_emitted = False


def _load_stats() -> Dict[str, Any]:
    global _missing_warning_emitted
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text())
        except Exception as exc:
            logger.warning(
                "Failed to parse strategy stats file %s: %s", STATS_FILE, exc
            )
            return {}
    if not _missing_warning_emitted:
        logger.warning("Strategy stats file %s not found", STATS_FILE)
        _missing_warning_emitted = True
    return {}


def get_expected_value(strategy: str) -> float:
    """Return expected value for ``strategy`` based on historical stats."""
    data = _load_stats().get(strategy, {})
    win_rate = float(data.get("win_rate", 0.0))
    avg_win = float(data.get("avg_win", 0.0))
    avg_loss = float(data.get("avg_loss", 0.0))
    return win_rate * avg_win + (1 - win_rate) * avg_loss
