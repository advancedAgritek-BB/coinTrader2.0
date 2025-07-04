from typing import Callable, Tuple, Dict, Iterable

import pandas as pd

from crypto_bot.utils.logger import setup_logger
from crypto_bot.signals.signal_fusion import SignalFusionEngine
from crypto_bot.strategy import (
    trend_bot,
    grid_bot,
    sniper_bot,
    dex_scalper,
    mean_bot,
    breakout_bot,
)

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")

STRATEGY_MAP: Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]] = {
    "trending": trend_bot.generate_signal,
    "sideways": grid_bot.generate_signal,
    "mean-reverting": mean_bot.generate_signal,
    "breakout": breakout_bot.generate_signal,
    "volatile": sniper_bot.generate_signal,
}


def strategy_for(regime: str) -> Callable[[pd.DataFrame], Tuple[float, str]]:
    """Return strategy callable for a given regime."""
    return STRATEGY_MAP.get(regime, grid_bot.generate_signal)


def strategy_name(regime: str, mode: str) -> str:
    """Return the name of the strategy for given regime and mode."""
    if mode == "cex":
        return "trend" if regime == "trending" else "grid"
    if mode == "onchain":
        return "sniper" if regime in {"breakout", "volatile"} else "dex_scalper"
    if regime == "trending":
        return "trend"
    if regime in {"breakout", "volatile"}:
        return "sniper"
    return "grid"


def route(
    regime: str,
    mode: str,
    config: Dict | None = None,
) -> Callable[[pd.DataFrame], Tuple[float, str]]:
    """Select a strategy based on market regime and operating mode.

    Parameters
    ----------
    regime : str
        Current market regime as classified by indicators.
    mode : str
        Trading environment, either ``cex``, ``onchain`` or ``auto``.
    config : dict | None
        Optional configuration dictionary. When ``meta_selector.enabled`` is
        ``True`` the strategy choice is delegated to the meta selector.

    Returns
    -------
    Callable[[pd.DataFrame], Tuple[float, str]]
        Strategy function returning a score and trade direction.
    """
    if mode == "onchain":
        if regime in {"breakout", "volatile"}:
            logger.info("Routing to sniper bot (onchain)")
            return sniper_bot.generate_signal
        logger.info("Routing to DEX scalper (onchain)")
        return dex_scalper.generate_signal


    if config and config.get("meta_selector", {}).get("enabled"):
        from . import meta_selector

        strategy_fn = meta_selector.choose_best(regime)
        logger.info("Meta selector chose %s for %s", strategy_fn.__name__, regime)
        return strategy_fn

    if config and config.get("signal_fusion", {}).get("enabled"):
        from . import meta_selector

        pairs_conf = config.get("signal_fusion", {}).get("strategies", [])
        mapping = getattr(meta_selector, "_STRATEGY_FN_MAP", {})
        strategies: list[tuple[Callable[[pd.DataFrame], Tuple[float, str]], float]] = []
        for name, weight in pairs_conf:
            fn = mapping.get(name)
            if fn:
                strategies.append((fn, float(weight)))
        if not strategies:
            strategies.append((strategy_for(regime), 1.0))
        engine = SignalFusionEngine(strategies)

        def fused(df: pd.DataFrame, cfg=None):
            return engine.fuse(df, cfg)

        logger.info("Routing to signal fusion engine")
        return fused

    strategy_fn = strategy_for(regime)
    logger.info("Routing to %s (%s)", strategy_fn.__name__, mode)
    return strategy_fn
