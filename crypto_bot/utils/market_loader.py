from typing import Iterable, List, Dict
import asyncio


async def load_kraken_symbols(
    exchange, exclude: Iterable[str] | None = None
) -> List[str]:
    """Return a list of active trading pairs on Kraken.

    Parameters
    ----------
    exchange : ccxt Exchange
        Exchange instance connected to Kraken.
    exclude : Iterable[str] | None
        Symbols to exclude from the result.
    """

    exclude_set = set(exclude or [])

    if asyncio.iscoroutinefunction(getattr(exchange, "load_markets", None)):
        markets = await exchange.load_markets()
    else:
        markets = await asyncio.to_thread(exchange.load_markets)

    symbols: List[str] = []
    for symbol, data in markets.items():
        if not data.get("active", True):
            continue
        if symbol in exclude_set:
            continue
        symbols.append(symbol)
    return symbols


async def fetch_ohlcv_async(
    exchange,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
) -> list | Exception:
    """Return OHLCV data for ``symbol`` using async I/O."""
    try:
        if use_websocket and hasattr(exchange, "watch_ohlcv"):
            data = await exchange.watch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if (
                limit
                and len(data) < limit
                and not force_websocket_history
                and hasattr(exchange, "fetch_ohlcv")
            ):
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
                    return await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                return await asyncio.to_thread(
                    exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit
                )
            return data
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
            return await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return await asyncio.to_thread(
            exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit
        )
    except Exception as exc:  # pragma: no cover - network
        return exc


async def load_ohlcv_parallel(
    exchange,
    symbols: Iterable[str],
    timeframe: str = "1h",
    limit: int = 100,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
) -> Dict[str, list]:
    """Fetch OHLCV data for multiple symbols concurrently."""

    tasks = [
        fetch_ohlcv_async(
            exchange,
            s,
            timeframe,
            limit,
            use_websocket,
            force_websocket_history,
        )
        for s in symbols
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    data: Dict[str, list] = {}
    for sym, res in zip(symbols, results):
        if isinstance(res, Exception):
            continue
        if res and len(res[0]) > 6:
            res = [[c[0], c[1], c[2], c[3], c[4], c[6]] for c in res]
        data[sym] = res
    return data
