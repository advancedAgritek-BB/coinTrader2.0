"""Kraken exchange client with simple request rate tracking."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from crypto_bot.utils.telemetry import telemetry
from crypto_bot.utils.metrics_logger import log_metrics_to_csv
from crypto_bot.utils.logger import LOG_DIR


class KrakenClient:
    """Thin wrapper around a ccxt Kraken exchange with IOPS metrics."""

    def __init__(self, exchange: Any) -> None:
        self._exchange = exchange
        self._timestamps: deque[float] = deque()
        self._emit_every = 300.0  # seconds
        self._last_emit = time.time()

    # delegate unknown attributes to the underlying exchange
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._exchange, name)

    def request(
        self,
        path: str,
        api: Any = "public",
        method: str = "GET",
        params: dict | None = None,
        headers: Any = None,
        body: Any = None,
        config: dict | None = None,
    ) -> Any:
        """Proxy request recording call timestamps for rate metrics."""

        params = params or {}
        config = config or {}

        result = self._exchange.request(path, api, method, params, headers, body, config)

        now = time.time()
        self._timestamps.append(now)
        cutoff = now - 300.0
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

        telemetry.inc("kraken.requests")

        if now - self._last_emit >= self._emit_every:
            rps = len(self._timestamps) / 300.0
            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kraken_iops": rps,
            }
            log_metrics_to_csv(metrics, str(LOG_DIR / "kraken_iops.csv"))
            self._last_emit = now

        return result
