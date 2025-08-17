from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, Iterable, Tuple

import pandas as pd

log = logging.getLogger(__name__)

# Optional OHLCV DataFrame provider supplied by the application. Strategies can
# request recent price data via :func:`get_ohlcv_df`.
_OHLCV_PROVIDER: Callable[[str, str, int], pd.DataFrame] | None = None


def set_ohlcv_provider(provider: Callable[[str, str, int], pd.DataFrame]) -> None:
    """Register a function used to fetch OHLCV dataframes."""

    global _OHLCV_PROVIDER
    _OHLCV_PROVIDER = provider


def get_ohlcv_df(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Return OHLCV data for ``symbol`` using the registered provider."""

    if _OHLCV_PROVIDER is None:
        raise RuntimeError("OHLCV provider not configured")
    return _OHLCV_PROVIDER(symbol, timeframe, limit)


def _df(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame | None:
    if _OHLCV_PROVIDER is None:
        return None
    try:
        return _OHLCV_PROVIDER(symbol, timeframe, limit)
    except Exception as e:  # pragma: no cover - defensive
        log.debug("OHLCV provider failed for %s %s: %s", symbol, timeframe, e)
        return None


async def score(
    mod: Any,
    symbols: Iterable[str] | None = None,
    timeframes: Iterable[str] | None = None,
) -> Dict[Tuple[str, str], Any]:
    """Compatibility adapter for legacy strategy modules.

    Supports three flavours of strategies:

    * Native ``score(symbols, timeframes)`` functions.
    * ``generate_signal(df, **kwargs)`` single-asset strategies.
    * ``generate_signal(df_a, df_b, **kwargs)`` pair/stat-arb strategies.

    Returns a dictionary keyed by ``(symbol or "A|B", timeframe)`` mapping to the
    produced signal or score.
    """

    symbols = list(symbols or [])
    timeframes = list(timeframes or [])

    # Prefer a native vectorised score implementation if provided.
    native = getattr(mod, "score", None)
    if callable(native) and native is not score:
        try:
            return await _maybe_await(native, symbols=symbols, timeframes=timeframes)
        except Exception:  # pragma: no cover - defensive
            log.exception("Strategy %s score() failed", getattr(mod, "__name__", mod))
            return {}

    gen = getattr(mod, "generate_signal", None)
    if not callable(gen):
        log.warning(
            "Strategy %s has neither score() nor generate_signal()",
            getattr(mod, "__name__", mod),
        )
        return {}

    sig = inspect.signature(gen).parameters
    out: Dict[Tuple[str, str], Any] = {}

    # Pair / stat-arb style strategies expect two dataframes.
    if "df_a" in sig and "df_b" in sig:
        pairs = getattr(mod, "PAIRS", None) or getattr(mod, "pairs", None) or []
        for a, b in pairs:
            for tf in timeframes:
                df_a = _df(a, tf)
                df_b = _df(b, tf)
                if df_a is None or df_b is None:
                    continue
                try:
                    res = gen(
                        df_a=df_a,
                        df_b=df_b,
                        symbol_a=a,
                        symbol_b=b,
                        timeframe=tf,
                    )
                except TypeError:
                    res = gen(df_a, df_b)
                if res:
                    out[(f"{a}|{b}", tf)] = res
        return out

    # Single-asset strategies
    for sym in symbols:
        for tf in timeframes:
            df = _df(sym, tf)
            if df is None:
                continue
            try:
                res = gen(df=df, symbol=sym, timeframe=tf)
            except TypeError:
                res = gen(df)
            if res:
                out[(sym, tf)] = res
    return out


def _maybe_await(func: Callable[..., Any] | None, *args, **kwargs) -> Any:
    """Invoke ``func`` which may be synchronous or asynchronous.

    We filter keyword arguments by the function signature but never fabricate
    positional arguments.
    """

    import asyncio

    if func is None:
        return {}
    try:
        sig = inspect.signature(func)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
        if inspect.iscoroutinefunction(func):
            return asyncio.create_task(func(*args, **allowed))
        return func(*args, **allowed)
    except Exception:  # pragma: no cover - defensive
        log.exception("Strategy call failed")
        return {}


__all__ = ["score", "set_ohlcv_provider", "get_ohlcv_df"]

