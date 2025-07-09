"""Pool watcher utilities for Solana."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator, Optional


@dataclass
class NewPoolEvent:
    """Event emitted when a new liquidity pool is detected."""

    pool_address: str
    token_mint: str
    creator: str
    liquidity: float
    tx_count: int = 0
    freeze_authority: str = ""
    mint_authority: str = ""
    timestamp: float = 0.0


class PoolWatcher:
    """Async engine that polls for new pools and yields :class:`NewPoolEvent`."""

    def __init__(self, url: str, interval: float = 5.0) -> None:
        self.url = url
        self.interval = interval
        self._running = False

    async def watch(self) -> AsyncGenerator[NewPoolEvent, None]:
        """Yield :class:`NewPoolEvent` objects as they are discovered."""
        self._running = True
        while self._running:
            await asyncio.sleep(self.interval)
            # Placeholder event - in a real implementation this would query an
            # RPC or indexer for pool creations.
            yield NewPoolEvent(
                pool_address="DEMO",
                token_mint="DEMO",
                creator="DEMO",
                liquidity=0.0,
            )

    def stop(self) -> None:
        """Stop the watcher loop."""
        self._running = False
