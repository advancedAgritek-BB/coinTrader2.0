import json
from pathlib import Path
from typing import Any, Dict

from .logger import LOG_DIR
from crypto_bot.selector import bandit


LOG_FILE = LOG_DIR / "strategy_performance.json"


def log_performance(record: Dict[str, Any]) -> None:
    """Append a trade result to ``strategy_performance.json``.

    The JSON file groups records by market regime and strategy::

        {
            "trending": {
                "trend_bot": [
                    {
                        "symbol": "BTC/USDT",
                        "pnl": 1.2,
                        "entry_time": "2024-01-01T00:00:00Z",
                        "exit_time": "2024-01-01T02:00:00Z"
                    }
                ]
            }
        }
    """
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
    bandit.update(
        record.get("symbol", ""),
        record.get("strategy", ""),
        float(record.get("pnl", 0.0)) > 0,
    )
