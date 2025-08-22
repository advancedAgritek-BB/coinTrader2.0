"""Centralized asyncio queues used across the trading bot."""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict


class TradeCandidate(TypedDict):
    """Represents a potential trade awaiting execution."""

    symbol: str
    side: str
    score: float
    strategy: str
    timeframe: str
    meta: dict[str, Any]


# Global queue of trade candidates awaiting execution
trade_queue: asyncio.Queue[TradeCandidate] = asyncio.Queue(maxsize=1000)
