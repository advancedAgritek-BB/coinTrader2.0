import json
from pathlib import Path
from typing import Any, Dict

STATS_FILE = Path("crypto_bot/logs/strategy_stats.json")


def _load_stats() -> Dict[str, Any]:
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text())
        except Exception:
            return {}
    return {}


def get_expected_value(strategy: str) -> float:
    """Return expected value for ``strategy`` based on historical stats."""
    data = _load_stats().get(strategy, {})
    win_rate = float(data.get("win_rate", 0.0))
    avg_win = float(data.get("avg_win", 0.0))
    avg_loss = float(data.get("avg_loss", 0.0))
    return win_rate * avg_win + (1 - win_rate) * avg_loss
