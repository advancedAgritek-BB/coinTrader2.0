from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, Tuple

import pandas as pd

logger = logging.getLogger(__name__)
log = logger


def _filter_kwargs(func, **kwargs):
    """Return only kwargs that the callable accepts (unless it has **kwargs)."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        # If we can't introspect, best effort: pass only 'df'
        return {k: v for k, v in kwargs.items() if k == "df"}
    accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_varkw:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


async def _invoke_strategy(gen, **ctx):
    """Call a strategy, awaiting if necessary, after filtering kwargs."""
    filtered = _filter_kwargs(gen, **ctx)
    # Handle pure async strategy functions
    if inspect.iscoroutinefunction(gen):
        return await gen(**filtered)
    # Handle sync functions that might still return an awaitable
    res = gen(**filtered)
    if inspect.isawaitable(res):
        return await res
    return res


def _normalize_result(val):
    """Normalize strategy outputs to (score: float, action: str)."""
    if val is None:
        return 0.0, "none"
    if isinstance(val, (tuple, list)) and len(val) == 2:
        try:
            return float(val[0]), str(val[1])
        except Exception:
            return 0.0, "none"
    if isinstance(val, (int, float)):
        return float(val), "none"
    logger.warning("Unexpected strategy result type: %r", val)
    return 0.0, "none"

# Optional OHLCV DataFrame provider supplied by the application. Strategies can
# request recent price data via :func:`get_ohlcv_df`.
_OHLCV_PROVIDER: Callable[[str, str, int], Awaitable[pd.DataFrame]] | None = None


def set_ohlcv_provider(
    provider: Callable[[str, str, int], Awaitable[pd.DataFrame]]
) -> None:
    """Register a function used to fetch OHLCV dataframes."""

    global _OHLCV_PROVIDER
    _OHLCV_PROVIDER = provider


async def get_ohlcv_df(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Return OHLCV data for ``symbol`` using the registered provider."""

    if _OHLCV_PROVIDER is None:
        raise RuntimeError("OHLCV provider not configured")
    return await _OHLCV_PROVIDER(symbol, timeframe, limit)


async def _df(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame | None:
    if _OHLCV_PROVIDER is None:
        return None
    try:
        return await _OHLCV_PROVIDER(symbol, timeframe, limit)
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

    name = getattr(mod, "__name__", mod)
    sig = inspect.signature(gen).parameters
    params = set(sig.keys())
    out: Dict[Tuple[str, str], Any] = {}

    # Pair / stat-arb style strategies expect two dataframes.
    if {"df_a", "df_b"} <= params:
        pairs = getattr(mod, "PAIRS", None) or getattr(mod, "pairs", None) or []
        for a, b in pairs:
            for tf in timeframes:
                df_a = await _df(a, tf)
                df_b = await _df(b, tf)
                if df_a is None or df_b is None:
                    continue
                try:
                    res = await _invoke_strategy(
                        gen,
                        df_a=df_a,
                        df_b=df_b,
                        symbol_a=a,
                        symbol_b=b,
                        timeframe=tf,
                    )
                except Exception:
                    logger.exception(
                        "Strategy %s scoring failed for %s|%s @ %s",
                        name,
                        a,
                        b,
                        tf,
                    )
                    res = (0.0, "none")
                score_action = _normalize_result(res)
                out[(f"{a}|{b}", tf)] = score_action
        return out

    # Single-asset strategies
    for sym in symbols:
        for tf in timeframes:
            df = await _df(sym, tf)
            if df is None:
                continue
            if {"df_a", "df_b"} & params:
                logger.debug(
                    "Skipping pairwise strategy %s for single-symbol pass", name
                )
                continue
            try:
                res = await _invoke_strategy(
                    gen, df=df, symbol=sym, timeframe=tf
                )
            except Exception:
                logger.exception(
                    "Strategy %s scoring failed for %s @ %s", name, sym, tf
                )
                res = (0.0, "none")
            score_action = _normalize_result(res)
            out[(sym, tf)] = score_action
    return out


def _maybe_await(func: Callable[..., Any] | None, *args, **kwargs) -> Any:
    """Invoke ``func`` which may be synchronous or asynchronous.

    We filter keyword arguments by the function signature but never fabricate
    positional arguments.
    """

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

