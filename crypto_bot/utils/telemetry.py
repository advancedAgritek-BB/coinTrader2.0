from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict

from .logger import LOG_DIR
from .metrics_logger import log_metrics_to_csv

LOG_FILE = LOG_DIR / "telemetry.csv"


class Telemetry:
    """Collect simple counter metrics during runtime."""

    def __init__(self) -> None:
        self._counters: Dict[str, int] = defaultdict(int)

    def inc(self, name: str, value: int = 1) -> None:
        """Increment ``name`` by ``value``."""
        self._counters[name] += value

    def snapshot(self) -> Dict[str, int]:
        """Return a copy of all counters."""
        return dict(self._counters)

    def reset(self) -> None:
        """Reset all counters."""
        self._counters.clear()

    def export_csv(self, path: str | Path | None = None) -> None:
        """Append current counters to a CSV file."""
        file = Path(path or LOG_FILE)
        metrics = {**self.snapshot(), "timestamp": datetime.utcnow().isoformat()}
        log_metrics_to_csv(metrics, str(file))


telemetry = Telemetry()
