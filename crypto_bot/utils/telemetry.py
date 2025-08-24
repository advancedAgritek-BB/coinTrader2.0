from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from prometheus_client import Counter

from .logger import LOG_DIR, setup_logger
from .metrics_logger import log_metrics_to_csv

LOG_FILE = LOG_DIR / "telemetry.csv"
logger = setup_logger(__name__, LOG_DIR / "telemetry.log")

# Prometheus counters keyed by telemetry name
PROM_COUNTERS: Dict[str, Counter] = {
    "analysis.skipped_no_df": Counter(
        "analysis_skipped_no_df",
        "Number of symbols skipped due to missing OHLCV data",
    ),
    "analysis.skipped_short_data": Counter(
        "analysis_skipped_short_data",
        "Number of symbols skipped due to insufficient OHLCV data length",
    ),
    "scan.ws_errors": Counter(
        "scan_ws_errors",
        "Number of WebSocket errors encountered while scanning",
    ),
    "ml_fallbacks": Counter(
        "ml_fallbacks",
        "Number of times ML fallback triggered",
    ),
    "analysis.regime_unknown": Counter(
        "analysis_regime_unknown",
        "Number of regime classifications returning unknown due to missing data",
    ),
    "analysis.regime_unknown_alerts": Counter(
        "analysis_regime_unknown_alerts",
        "Number of alerts triggered for sustained unknown regime classifications",
    ),
    "signals.missing_ohlcv": Counter(
        "signals_missing_ohlcv",
        "Number of symbols skipped due to missing OHLCV data during signal generation",
    ),
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


def dump() -> str:
    """Log current counters and return formatted string."""
    snap = telemetry.snapshot()
    msg = ", ".join(f"{k}: {v}" for k, v in sorted(snap.items()))
    logger.info("Telemetry snapshot - %s", msg)
    return msg
