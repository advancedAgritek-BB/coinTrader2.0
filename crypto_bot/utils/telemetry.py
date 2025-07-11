from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from prometheus_client import Counter

from .logger import LOG_DIR
from .metrics_logger import log_metrics_to_csv

LOG_FILE = LOG_DIR / "telemetry.csv"

# Prometheus counters keyed by telemetry name
PROM_COUNTERS: Dict[str, Counter] = {
    "analysis.skipped_no_df": Counter(
        "analysis_skipped_no_df",
        "Number of symbols skipped due to missing OHLCV data",
    )
}


class Telemetry:
    """Collect simple counter metrics during runtime."""

    def __init__(self) -> None:
        self._counters: Dict[str, int] = defaultdict(int)

    def inc(self, name: str, value: int = 1) -> None:
        """Increment ``name`` by ``value``."""
        self._counters[name] += value
        counter = PROM_COUNTERS.get(name)
        if counter is not None:
            counter.inc(value)

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


def write_cycle_metrics(metrics: Dict[str, Any], cfg: Dict) -> None:
    """Write cycle metrics and export telemetry counters.

    Parameters
    ----------
    metrics:
        Mapping of metric names to values.
    cfg:
        Bot configuration dictionary providing metric settings.
    """
    if cfg.get("metrics_enabled") and cfg.get("metrics_backend") == "csv":
        log_metrics_to_csv(
            metrics,
            cfg.get("metrics_output_file", str(LOG_DIR / "metrics.csv")),
        )
        telemetry.export_csv(
            cfg.get("metrics_output_file", str(LOG_DIR / "telemetry.csv"))
        )


