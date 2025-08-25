from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Any, Awaitable, Callable, Deque

import ccxt.async_support as ccxt  # type: ignore
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from crypto_bot.utils.logger import LOG_DIR, setup_logger

DDoSProtection = getattr(ccxt, "DDoSProtection", Exception)
RateLimitExceeded = getattr(ccxt, "RateLimitExceeded", Exception)

DEFAULT_MAX_CONCURRENCY = int(os.getenv("KRAKEN_MAX_CONCURRENCY", "5"))
_max_concurrency = DEFAULT_MAX_CONCURRENCY
semaphore = asyncio.Semaphore(_max_concurrency)

_REQUEST_WINDOW = 300.0  # 5 minutes

logger = setup_logger(__name__, LOG_DIR / "kraken_client.log")


def _build_session(pool_maxsize: int = 100, retries: int = 3) -> requests.Session:
    """Return a :class:`requests.Session` with an expanded adapter pool."""

    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=0.3, raise_on_status=False)
    adapter = HTTPAdapter(
        pool_connections=pool_maxsize,
        pool_maxsize=pool_maxsize,
        max_retries=retry,
    )
    session.mount("https://api.kraken.com", adapter)
    session.mount("http://api.kraken.com", adapter)
    session.headers.update({"User-Agent": "coinTrader2.0"})
    return session


class KrakenClient:
    """Asynchronous Kraken client with shared rate limiting and statistics."""

    def __init__(self, exchange: "ccxt.Exchange") -> None:
        self._exchange = exchange
        self._times: Deque[float] = deque()

    async def _call_with_retries(
        self, method: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        delay = 1.0
        for attempt in range(5):
            try:
                async with semaphore:
                    result = await method(*args, **kwargs)
                self._record_request()
                return result
            except (DDoSProtection, RateLimitExceeded) as exc:
                if attempt == 4:
                    raise
                logger.warning(
                    "Kraken rate limited: %s; retrying in %.1fs", exc, delay
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 32.0)
            except asyncio.CancelledError:
                raise

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._exchange, name)
        if not callable(attr):
            return attr

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self._call_with_retries(attr, *args, **kwargs)

        return wrapper

    def _record_request(self) -> None:
        now = time.monotonic()
        self._times.append(now)
        self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - _REQUEST_WINDOW
        while self._times and self._times[0] < cutoff:
            self._times.popleft()

    @property
    def avg_request_rate(self) -> float:
        """Average number of requests per minute over the last 5 minutes."""
        now = time.monotonic()
        self._trim(now)
        return len(self._times) / (_REQUEST_WINDOW / 60.0)

    async def list_markets(self) -> dict[str, Any]:
        """Fetch and return market metadata keyed by symbol."""

        loader = getattr(self._exchange, "load_markets", None)
        if callable(loader):
            markets = await self._call_with_retries(loader)
        else:
            fetcher = getattr(self._exchange, "fetch_markets")
            markets = await self._call_with_retries(fetcher)
        if isinstance(markets, list):
            return {m.get("symbol", m): m for m in markets}
        return markets

    async def close(self) -> None:
        await self._exchange.close()


_client: KrakenClient | None = None


def get_kraken_client(
    *,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    pool_maxsize: int | None = None,
    **exchange_kwargs: Any,
) -> KrakenClient:
    """Return a shared :class:`KrakenClient` instance.

    The first call may specify ``max_concurrency`` and keyword arguments passed to
    ``ccxt.kraken``. ``pool_maxsize`` controls the underlying HTTP connection pool.
    Subsequent calls ignore these parameters and return the same instance.
    """
    global _client, semaphore, _max_concurrency
    if _client is None:
        _max_concurrency = max_concurrency
        semaphore = asyncio.Semaphore(_max_concurrency)
        params = {"enableRateLimit": True}
        if pool_maxsize is None:
            pool_maxsize = int(os.getenv("HTTP_POOL_SIZE", "50"))
        params["session"] = _build_session(pool_maxsize=pool_maxsize)
        params.update(exchange_kwargs)
        exchange = ccxt.kraken(params)
        _client = KrakenClient(exchange)
    return _client


__all__ = ["KrakenClient", "get_kraken_client", "semaphore"]
