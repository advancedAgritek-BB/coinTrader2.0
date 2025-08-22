import asyncio
import aiohttp
from typing import Optional

_session: Optional[aiohttp.ClientSession] = None


def get_session() -> aiohttp.ClientSession:
    """Return a shared :class:`aiohttp.ClientSession` tied to the current loop."""
    global _session
    loop = asyncio.get_event_loop()
    if (
        _session is None
        or _session.closed
        or getattr(_session, "_loop", loop) is not loop
    ):
        connector = aiohttp.TCPConnector(limit=100)
        _session = aiohttp.ClientSession(connector=connector)
    return _session


async def close_session() -> None:
    """Close the shared :class:`aiohttp.ClientSession` if it exists."""
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None
