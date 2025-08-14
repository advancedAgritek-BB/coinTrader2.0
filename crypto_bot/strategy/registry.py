import importlib
import logging
from types import ModuleType
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)


def load_from_config(config: dict) -> List[ModuleType]:
    """Return strategy modules referenced in ``config``.

    Strategy names are gathered from the ``strategy_router.regimes`` section and
    imported from :mod:`crypto_bot.strategy`.
    """
    names: set[str] = set()
    router = config.get("strategy_router", {}) or {}
    for lst in router.get("regimes", {}).values():
        if isinstance(lst, str):
            names.add(lst)
        else:
            try:
                names.update(str(n) for n in lst)
            except Exception:
                continue
    strategies: List[ModuleType] = []
    for name in names:
        try:
            strategies.append(importlib.import_module(f"crypto_bot.strategy.{name}"))
        except Exception:  # pragma: no cover - optional strategies
            logger.warning("Failed to import strategy %s", name)
    return strategies


def compute_required_lookback_per_tf(strategies: Iterable[ModuleType]) -> Dict[str, int]:
    """Return per-timeframe max lookback required by ``strategies``."""
    req: Dict[str, int] = {}
    for strat in strategies:
        lr = getattr(strat, "required_lookback", None)
        if not callable(lr):
            continue
        try:
            for tf, n in (lr() or {}).items():
                req[tf] = max(req.get(tf, 0), int(n))
        except Exception:  # pragma: no cover - ignore bad implementations
            continue
    return req


def filter_by_warmup(config: dict, strategies: List[ModuleType]) -> List[ModuleType]:
    """Filter ``strategies`` based on available warmup candles.

    If ``data.auto_raise_warmup`` is enabled missing warmup candles are
    automatically bumped; otherwise strategies requiring more history are
    removed and a warning is logged.
    """
    warmup_map = config.get("warmup_candles", {}) or {}
    auto_raise = bool(config.get("data", {}).get("auto_raise_warmup", False))
    if auto_raise:
        required = compute_required_lookback_per_tf(strategies)
        for tf, need in required.items():
            have = int(warmup_map.get(tf, 0) or 0)
            if have < need:
                logger.info(
                    "Auto-raising warmup_candles[%s] %d -> %d to satisfy strategies.",
                    tf,
                    have,
                    need,
                )
                warmup_map[tf] = need
    allowed: List[ModuleType] = []
    disabled: List[str] = []
    for strat in strategies:
        lr = getattr(strat, "required_lookback", None)
        needs = lr() if callable(lr) else {}
        ok = True
        for tf, need in needs.items():
            have = int(warmup_map.get(tf, 0) or 0)
            if have < int(need):
                ok = False
                disabled.append(f"{strat.__name__}[{tf}]")
                break
        if ok:
            allowed.append(strat)
    if disabled and not auto_raise:
        logger.warning(
            "Insufficient warmup_candles; disabling: %s", ", ".join(disabled)
        )
    return allowed


def load_enabled(config: dict) -> List[ModuleType]:
    """Load strategies from ``config`` applying warmup guards."""
    strategies = load_from_config(config)
    return filter_by_warmup(config, strategies)

