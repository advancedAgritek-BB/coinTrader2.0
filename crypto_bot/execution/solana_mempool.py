"""Utility to inspect the Solana mempool for high priority fees.

This module provides a helper that queries a priority fee API to gauge
network congestion. It falls back to the environment variable
``MOCK_PRIORITY_FEE`` so tests and offline runs can control the value.
"""

from __future__ import annotations

import os
import asyncio
from collections import deque
from typing import Optional, Deque

import aiohttp


class SolanaMempoolMonitor:
    """Simple monitor for Solana priority fees and volume."""

    def __init__(
        self,
        priority_fee_url: Optional[str] = None,
        volume_url: Optional[str] = None,
        *,
        history_size: int = 20,
    ) -> None:
        self.priority_fee_url = priority_fee_url or os.getenv(
            "SOLANA_PRIORITY_FEE_URL",
            "https://mempool.solana.com/api/v0/fees/priority_fee",
        )
        # Use the same endpoint for volume by default. Tests can override via
        # ``SOLANA_VOLUME_URL`` or by passing ``volume_url`` explicitly.
        self.volume_url = volume_url or os.getenv("SOLANA_VOLUME_URL", self.priority_fee_url)
        self._volume_history: Deque[float] = deque(maxlen=history_size)
        self._volume_task: asyncio.Task | None = None

    async def fetch_priority_fee(self) -> float:
        """Return the current priority fee per compute unit in micro lamports."""
        mock_fee = os.getenv("MOCK_PRIORITY_FEE")
        if mock_fee is not None:
            try:
                return float(mock_fee)
            except ValueError:
                return 0.0
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.priority_fee_url, timeout=5) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    if isinstance(data, dict):
                        return float(data.get("priorityFee", 0.0))
        except Exception:
            pass
        return 0.0

    async def is_suspicious(self, threshold: float) -> bool:
        """Return True when the priority fee exceeds ``threshold``."""
        fee = await self.fetch_priority_fee()
        return fee >= threshold

    async def _fetch_volume(self) -> float:
        """Return the latest observed transaction volume from ``self.volume_url``."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.volume_url, timeout=5) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    if isinstance(data, dict):
                        # Support multiple field names for flexibility.
                        vol = (
                            data.get("volume")
                            or data.get("recentVolume")
                            or data.get("txVolume")
                        )
                        if vol is not None:
                            return float(vol)
        except Exception:
            pass
        return 0.0

    async def _volume_loop(self, interval: float) -> None:
        """Background task that periodically records volume."""
        while True:
            vol = await self._fetch_volume()
            self._volume_history.append(vol)
            await asyncio.sleep(interval)

    def start_volume_collection(self, interval: float = 60.0) -> None:
        """Start a background task updating the volume history."""
        if self._volume_task is None or self._volume_task.done():
            self._volume_task = asyncio.create_task(self._volume_loop(interval))

    def stop_volume_collection(self) -> None:
        """Stop the background volume collection task."""
        if self._volume_task and not self._volume_task.done():
            self._volume_task.cancel()

    async def get_recent_volume(self) -> float:
        """Return the most recent volume reading, fetching if necessary."""
        if not self._volume_history:
            vol = await self._fetch_volume()
            self._volume_history.append(vol)
        return self._volume_history[-1] if self._volume_history else 0.0

    async def get_average_volume(self) -> float:
        """Return the average of the recorded volume history."""
        if not self._volume_history:
            await self.get_recent_volume()
        if not self._volume_history:
            return 0.0
        return sum(self._volume_history) / len(self._volume_history)
