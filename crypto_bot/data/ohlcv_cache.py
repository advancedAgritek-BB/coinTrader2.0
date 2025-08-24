import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from .locks import timeframe_lock


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


class OHLCVCache:
    """Manage OHLCV caching with per-timeframe locks."""

    def __init__(self, cfg: Any, logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    async def update_intraday(self, timeframes: list[str]):
        """Run per-timeframe updates concurrently."""
        self.logger.info("Updating OHLCV cache for timeframes: %s", timeframes)
        await asyncio.gather(*(self._update_tf(tf) for tf in timeframes))

    async def _update_tf(self, tf: str) -> None:
        lock = self._get_timeframe_lock(tf)
        if lock.locked():
            self.logger.info("Skip: %s update already running.", tf)
            return

        async with lock:
            self.logger.info("Starting OHLCV update for timeframe %s", tf)
            warmup = self.cfg.warmup_candles.get(tf)
            backfill_days = self.cfg.deep_backfill_days.get(tf) or self.cfg.backfill_days.get(tf)
            start = None
            if backfill_days:
                start = utc_now() - timedelta(days=backfill_days)
                self.logger.info(
                    "Clamping backfill for %s to %s days (%s)",
                    tf,
                    backfill_days,
                    start.isoformat(),
                )
            if warmup:
                self.logger.info("Ensuring warmup candles for %s: %s", tf, warmup)

            await self._fetch_and_store(tf, warmup=warmup, start=start)
            self.logger.info("Completed OHLCV update for timeframe %s", tf)

    def _get_timeframe_lock(self, tf: str) -> asyncio.Lock:
        """Return a unique lock for ``tf``."""
        return timeframe_lock(tf)

    async def _fetch_and_store(
        self, tf: str, *, warmup: int | None = None, start: datetime | None = None
    ) -> None:
        """Fetch and persist OHLCV data for *tf*.

        Subclasses should override this method with concrete implementation.
        """
        raise NotImplementedError
