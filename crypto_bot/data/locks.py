import asyncio
from typing import Dict

# Global dictionary of per-timeframe locks
TF_LOCKS: Dict[str, asyncio.Lock] = {}


def timeframe_lock(timeframe: str) -> asyncio.Lock:
    """Return an asyncio lock dedicated to the given timeframe."""
    return TF_LOCKS.setdefault(timeframe, asyncio.Lock())
