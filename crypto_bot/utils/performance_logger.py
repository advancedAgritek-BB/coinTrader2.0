import json
from pathlib import Path
from typing import Any, Dict

LOG_FILE = Path("crypto_bot/logs/strategy_performance.json")


def log_performance(record: Dict[str, Any]) -> None:
    """Append trade performance record grouped by regime and strategy."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {}
    if LOG_FILE.exists():
        try:
            data = json.loads(LOG_FILE.read_text())
        except Exception:
            data = {}
    regime = record.get("regime", "unknown")
    strategy = record.get("strategy", "unknown")
    data.setdefault(regime, {}).setdefault(strategy, []).append(
        {
            "symbol": record.get("symbol", ""),
            "pnl": record.get("pnl", 0.0),
            "entry_time": record.get("entry_time", ""),
            "exit_time": record.get("exit_time", ""),
        }
    )
    LOG_FILE.write_text(json.dumps(data))
