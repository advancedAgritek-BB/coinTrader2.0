from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def fetch_ohlcv_chunked(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    min_bars: int,
    limit: int = 500,
    rate_limit_ms: int | None = None,
):
    """Fetch OHLCV data in successive chunks until ``min_bars`` are gathered."""

    out: list = []
    last_since = since_ms
    rate = rate_limit_ms or getattr(exchange, "rateLimit", 0)
    while len(out) < min_bars:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=last_since, limit=limit)
        if not batch:
            break
        out.extend(batch)
        last_since = batch[-1][0] + 1  # avoid duplicates
        time.sleep(max(rate, 200) / 1000.0)
    return out[:min_bars]


def bootstrap_timeframe(
    exchange,
    symbol: str,
    timeframe: str,
    warmup_bars: int,
):
    """Bootstrap OHLCV history for ``symbol`` and ``timeframe``."""

    tf_ms = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }[timeframe]
    since_ms = int(
        (
            datetime.now(timezone.utc)
            - timedelta(milliseconds=tf_ms * warmup_bars)
        ).timestamp()
        * 1000
    )
    data = fetch_ohlcv_chunked(
        exchange,
        symbol,
        timeframe,
        since_ms,
        min_bars=warmup_bars,
        limit=500,
    )
    if len(data) < max(100, warmup_bars // 3):
        logger.warning("%s bootstrap for %s is short: %d bars", timeframe, symbol, len(data))
    return data
