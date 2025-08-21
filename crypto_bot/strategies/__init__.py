from __future__ import annotations

import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, Tuple

import pandas as pd

from crypto_bot.config import cfg
from crypto_bot.signals.normalize import normalize_strategy_result

logger = logging.getLogger(__name__)
log = logger


def _filter_kwargs(func: Callable, **kwargs) -> Dict[str, Any]:
    """Keep only kwargs the strategy function accepts."""
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


async def _maybe_await(res: Any) -> Any:
    """Await ``res`` if it is awaitable."""
    if inspect.isawaitable(res):
        return await res
    return res

async def run_strategy(
    gen: Callable,
    *,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    **kwargs,
) -> Dict[str, Any]:
    """Execute ``gen`` and normalise the result.

    The strategy is invoked with ``df``, ``symbol`` and ``timeframe``. Unexpected
    keyword arguments are filtered out before invocation. Any exception is
    logged and a neutral result with ``meta['reason']`` set is returned.
    """

    try:
        args = {"df": df, "symbol": symbol, "timeframe": timeframe, **kwargs}
        filtered = _filter_kwargs(gen, **args)
        res = gen(**filtered)
        res = await _maybe_await(res)
    except Exception:
        logger.exception(
            "Strategy %s execution failed", getattr(gen, "__name__", gen)
        )
        return {"score": 0.0, "signal": "none", "meta": {"reason": "exception"}}

    res = normalize_strategy_result(res)
    reason = res.get("meta", {}).get("reason")
    if reason:
        logger.debug("Strategy %s result %s (%s)", getattr(gen, "__name__", gen), res, reason)
    else:
        logger.debug("Strategy %s result %s", getattr(gen, "__name__", gen), res)
    return res


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

    Returns a dictionary keyed by ``(symbol or "A|B", timeframe)`` mapping to a
    result dict with ``score``, ``signal`` and ``meta`` keys.
    """

    symbols = list(symbols or [])
    timeframes = list(timeframes or [])

    min_needed = max(200, getattr(cfg, "indicator_lookback", 200))

    # Prefer a native vectorised score implementation if provided.
    native = getattr(mod, "score", None)
    if callable(native) and native is not score:
        try:
            return await _maybe_call(native, symbols=symbols, timeframes=timeframes)
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
                    params = inspect.signature(gen).parameters
                    filtered = {
                        "df_a": df_a.tail(min_needed),
                        "df_b": df_b.tail(min_needed),
                    }
                    if "symbol_a" in params:
                        filtered["symbol_a"] = a
                    if "symbol_b" in params:
                        filtered["symbol_b"] = b
                    if "timeframe" in params:
                        filtered["timeframe"] = tf
                    raw = await _maybe_await(gen(**filtered))
                    res = normalize_strategy_result(raw)
                except Exception:
                    logger.exception(
                        "Strategy %s scoring failed for %s|%s @ %s",
                        name,
                        a,
                        b,
                        tf,
                    )
                    res = {"score": 0.0, "signal": "none", "meta": {"reason": "exception"}}
                reason = res.get("meta", {}).get("reason")
                if reason:
                    logger.debug(
                        "%s %s|%s @ %s -> %s (%s)",
                        name,
                        a,
                        b,
                        tf,
                        res,
                        reason,
                    )
                else:
                    logger.debug(
                        "%s %s|%s @ %s -> %s",
                        name,
                        a,
                        b,
                        tf,
                        res,
                    )
                out[(f"{a}|{b}", tf)] = res
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
            res = await run_strategy(
                gen,
                df=df.tail(min_needed),
                symbol=sym,
                timeframe=tf,
            )
            out[(sym, tf)] = res
    return out


async def _maybe_call(func: Callable[..., Any] | None, *args, **kwargs) -> Any:
    """Invoke ``func`` which may be synchronous or asynchronous."""

    if func is None:
        return {}
    try:
        filtered = _filter_kwargs(func, **kwargs)
        res = func(*args, **filtered)
        return await _maybe_await(res)
    except Exception:  # pragma: no cover - defensive
        log.exception("Strategy call failed")
        return {}


__all__ = ["score", "set_ohlcv_provider", "get_ohlcv_df"]

