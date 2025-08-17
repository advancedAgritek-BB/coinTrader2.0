"""Strategy orchestration helpers.

This module provides a thin wrapper around :mod:`crypto_bot.strategies.loader`
that keeps track of instantiated strategies and exposes convenience helpers for
initialization and scoring.  The goal is to offer a simple interface for the
rest of the project while remaining backward compatible with the previous
``load_strategies`` export.
"""

from __future__ import annotations

from typing import Any, Callable

import inspect
import logging

from . import loader

logger = logging.getLogger(__name__)

# Module-level store for strategies that have been loaded/initialised.  The
# mapping keys are strategy names while the values are the instantiated objects.
LOADED_STRATEGIES: dict[str, Any] = {}


def _supports_mode(strategy: Any, mode: str) -> bool:
    """Return ``True`` if ``strategy`` declares compatibility with ``mode``.

    Strategies may optionally expose a ``mode`` attribute (single value) or a
    ``modes`` attribute (an iterable).  When neither is present, the strategy is
    assumed to support all modes.
    """

    modes = getattr(strategy, "modes", None)
    if modes is not None:
        if isinstance(modes, str):
            return modes == mode
        return mode in set(modes)
    s_mode = getattr(strategy, "mode", None)
    if s_mode is None:
        return True
    if isinstance(s_mode, str):
        return s_mode == mode
    return mode in set(s_mode)


async def _maybe_await(func: Callable[..., Any], *args, **kwargs) -> Any:
    """Call a strategy function with only the kwargs it accepts.

    Works with both sync and async strategy functions.
    """

    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}

    if inspect.iscoroutinefunction(func):
        return await func(*args, **filtered)
    return func(*args, **filtered)


async def initialize(symbols: list[str], mode: str = "cex") -> None:
    """Load strategies and run their ``initialize`` hooks if present.

    Parameters
    ----------
    symbols:
        Symbols that strategies should operate on.
    mode:
        Execution mode, e.g. ``"cex"`` or ``"onchain"``.
    """

    global LOADED_STRATEGIES
    LOADED_STRATEGIES = {}

    try:
        loaded = loader.load_strategies(mode)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Strategy loading failed")
        loaded = []

    for strategy in loaded:
        if not _supports_mode(strategy, mode):
            continue
        name = getattr(strategy, "name", strategy.__class__.__name__)
        init_hook = getattr(strategy, "initialize", None)
        if callable(init_hook):
            try:
                sig = inspect.signature(init_hook)
                if sig.parameters:
                    await _maybe_await(init_hook, symbols)
                else:  # pragma: no cover - rare case
                    await _maybe_await(init_hook)
            except Exception:
                logger.exception("Strategy %s failed to initialise", name)
                continue
        LOADED_STRATEGIES[name] = strategy


async def score(*, symbols: list[str], timeframes: list[str]) -> dict[str, float]:
    """Return a mapping of strategy name to score.

    Each loaded strategy is queried for a scoring function.  Supported attribute
    names are ``score`` and ``generate_signal``.  The first numerical component
    of the returned value is used as the score.
    """

    results: dict[str, float] = {}
    for name, strategy in LOADED_STRATEGIES.items():
        func: Callable[..., Any] | None = None
        for attr in ("score", "generate_signal"):
            func = getattr(strategy, attr, None)
            if callable(func):
                break
        if not func:
            continue
        try:
            res = await _maybe_await(func, symbols=symbols, timeframes=timeframes)
            if isinstance(res, tuple):
                res = res[0]
            results[name] = float(res)
        except Exception:
            logger.exception("Strategy %s scoring failed", name)
    return results


# Re-export for backwards compatibility
load_strategies = loader.load_strategies

__all__ = ["load_strategies", "initialize", "score", "LOADED_STRATEGIES"]

