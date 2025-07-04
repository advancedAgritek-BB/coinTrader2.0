from typing import Iterable, List, Dict
import asyncio
import inspect
import pandas as pd


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
    since: int | None = None,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
) -> list | Exception:
    """Return OHLCV data for ``symbol`` using async I/O."""
    try:
        if use_websocket and hasattr(exchange, "watch_ohlcv"):
            params = inspect.signature(exchange.watch_ohlcv).parameters
            kwargs = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
            if since is not None and "since" in params:
                kwargs["since"] = since
            data = await exchange.watch_ohlcv(**kwargs)
            if (
                limit
                and len(data) < limit
                and not force_websocket_history
                and hasattr(exchange, "fetch_ohlcv")
            ):
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
                    params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                    kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
                    if since is not None and "since" in params_f:
                        kwargs_f["since"] = since
                    return await exchange.fetch_ohlcv(**kwargs_f)
                params_f = inspect.signature(exchange.fetch_ohlcv).parameters
                kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
                if since is not None and "since" in params_f:
                    kwargs_f["since"] = since
                return await asyncio.to_thread(exchange.fetch_ohlcv, **kwargs_f)
            return data
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
            params_f = inspect.signature(exchange.fetch_ohlcv).parameters
            kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
            if since is not None and "since" in params_f:
                kwargs_f["since"] = since
            return await exchange.fetch_ohlcv(**kwargs_f)
        params_f = inspect.signature(exchange.fetch_ohlcv).parameters
        kwargs_f = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
        if since is not None and "since" in params_f:
            kwargs_f["since"] = since
        return await asyncio.to_thread(exchange.fetch_ohlcv, **kwargs_f)
    except Exception as exc:  # pragma: no cover - network
        return exc


async def load_ohlcv_parallel(
    exchange,
    symbols: Iterable[str],
    timeframe: str = "1h",
    limit: int = 100,
    since_map: Dict[str, int] | None = None,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    max_concurrent: int | None = None,
) -> Dict[str, list]:
    """Fetch OHLCV data for multiple symbols concurrently."""

    since_map = since_map or {}

    if max_concurrent is not None:
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer or None")
        sem = asyncio.Semaphore(max_concurrent)
    else:
        sem = None

    async def sem_fetch(sym: str):
        if sem:
            async with sem:
                return await fetch_ohlcv_async(
                    exchange,
                    sym,
                    timeframe=timeframe,
                    limit=limit,
                    since=since_map.get(sym),
                    use_websocket=use_websocket,
                    force_websocket_history=force_websocket_history,
                )
        return await fetch_ohlcv_async(
            exchange,
            sym,
            timeframe=timeframe,
            limit=limit,
            since=since_map.get(sym),
            use_websocket=use_websocket,
            force_websocket_history=force_websocket_history,
        )

    tasks = [asyncio.create_task(sem_fetch(s)) for s in symbols]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    data: Dict[str, list] = {}
    for sym, res in zip(symbols, results):
        if isinstance(res, Exception):
            continue
        if res and len(res[0]) > 6:
            res = [[c[0], c[1], c[2], c[3], c[4], c[6]] for c in res]
        data[sym] = res
    return data


async def update_ohlcv_cache(
    exchange,
    cache: Dict[str, pd.DataFrame],
    symbols: Iterable[str],
    timeframe: str = "1h",
    limit: int = 100,
    use_websocket: bool = False,
    force_websocket_history: bool = False,
    max_concurrent: int | None = None,
) -> Dict[str, pd.DataFrame]:
    """Update cached OHLCV DataFrames with new candles.

    Parameters
    ----------
    max_concurrent : int | None, optional
        Maximum number of concurrent OHLCV requests. ``None`` means no limit.
    """

    if max_concurrent is not None:
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer or None")

    since_map: Dict[str, int] = {}
    for sym in symbols:
        df = cache.get(sym)
        if df is not None and not df.empty:
            since_map[sym] = int(df["timestamp"].iloc[-1])

    data_map = await load_ohlcv_parallel(
        exchange,
        symbols,
        timeframe,
        limit,
        since_map,
        use_websocket,
        force_websocket_history,
        max_concurrent,
    )

    for sym in symbols:
        data = data_map.get(sym)
        if not data:
            full = await load_ohlcv_parallel(
                exchange,
                [sym],
                timeframe,
                limit,
                None,
                use_websocket,
                force_websocket_history,
                max_concurrent,
            )
            data = full.get(sym)
        if data is None:
            continue
        df_new = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        if sym in cache and not cache[sym].empty:
            last_ts = cache[sym]["timestamp"].iloc[-1]
            df_new = df_new[df_new["timestamp"] > last_ts]
            if df_new.empty:
                continue
            cache[sym] = pd.concat([cache[sym], df_new], ignore_index=True)
        else:
            cache[sym] = df_new
    return cache
