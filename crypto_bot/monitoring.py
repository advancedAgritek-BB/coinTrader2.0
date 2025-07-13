from __future__ import annotations

"""Helper functions for recording operational metrics."""

from datetime import datetime
from typing import Mapping

from .utils.metrics_logger import log_metrics_to_csv
from .utils.logger import LOG_DIR


def record_sol_scanner_metrics(tokens: int, latency: float, cfg: Mapping[str, object]) -> None:
    """Log Solana scanner metrics to the configured CSV backend."""
    if not cfg.get("metrics_enabled") or cfg.get("metrics_backend") != "csv":
        return
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "solana_scanner.tokens_per_scan": tokens,
        "solana_scanner.scanner_latency": latency,
    }
    output = str(cfg.get("metrics_output_file", LOG_DIR / "metrics.csv"))
    log_metrics_to_csv(metrics, output)
