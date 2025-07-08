from typing import Callable, Tuple, Dict, Iterable, Union

import pandas as pd

from pathlib import Path
import yaml

from crypto_bot.utils.logger import setup_logger
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.strategy import (
    trend_bot,
    grid_bot,
    sniper_bot,
    dex_scalper,
    mean_bot,
    breakout_bot,
    micro_scalp_bot,
    bounce_scalper,
)

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CONFIG_PATH) as f:
    DEFAULT_CONFIG = yaml.safe_load(f)

def get_strategy_by_name(name: str) -> Callable[[pd.DataFrame], Tuple[float, str]] | None:
    """Return strategy callable for ``name`` if available."""
    from . import meta_selector
    from .rl import strategy_selector as rl_selector

    mapping: Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]] = {}
    mapping.update(getattr(meta_selector, "_STRATEGY_FN_MAP", {}))
    mapping.update(getattr(rl_selector, "_STRATEGY_FN_MAP", {}))
    return mapping.get(name)


def _build_mappings(config: Dict) -> tuple[
    Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]],
    Dict[str, list[Callable[[pd.DataFrame], Tuple[float, str]]]],
]:
    """Return mapping dictionaries from configuration."""
    regimes = config.get("strategy_router", {}).get("regimes", {})
    strat_map: Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]] = {}
    regime_map: Dict[str, list[Callable[[pd.DataFrame], Tuple[float, str]]]] = {}
    for regime, names in regimes.items():
        if isinstance(names, str):
            names = [names]
        funcs = [get_strategy_by_name(n) for n in names]
        funcs = [f for f in funcs if f]
        if funcs:
            strat_map[regime] = funcs[0]
            regime_map[regime] = funcs
    return strat_map, regime_map


STRATEGY_MAP, REGIME_STRATEGIES = _build_mappings(DEFAULT_CONFIG)


def strategy_for(regime: str, config: Dict | None = None) -> Callable[[pd.DataFrame], Tuple[float, str]]:
    """Return strategy callable for a given regime."""
    mapping, _ = _build_mappings(config or DEFAULT_CONFIG)
    return mapping.get(regime, grid_bot.generate_signal)


def get_strategies_for_regime(regime: str, config: Dict | None = None) -> list[Callable[[pd.DataFrame], Tuple[float, str]]]:
    """Return list of strategies mapped to ``regime``."""
    _, mapping = _build_mappings(config or DEFAULT_CONFIG)
    return mapping.get(regime, [grid_bot.generate_signal])


def strategy_name(regime: str, mode: str) -> str:
    """Return the name of the strategy for given regime and mode."""
    if mode == "cex":
        return "trend" if regime == "trending" else "grid"
    if mode == "onchain":
        return "sniper" if regime in {"breakout", "volatile"} else "dex_scalper"
    if regime == "trending":
        return "trend"
    if regime == "scalp":
        return "micro_scalp"
    if regime in {"breakout", "volatile"}:
        return "sniper"
    return "grid"


def route(
    regime: Union[str, Dict[str, str]],
    mode: str,
    config: Dict | None = None,
    notifier: TelegramNotifier | None = None,
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
    notifier : TelegramNotifier | None
        Optional notifier used to send a message when the strategy is called.

    Returns
    -------
    Callable[[pd.DataFrame], Tuple[float, str]]
        Strategy function returning a score and trade direction.
    """
    def _wrap(fn: Callable[[pd.DataFrame], Tuple[float, str]]):
        if notifier is None:
            return fn

        def wrapped(df: pd.DataFrame, cfg=None):
            try:
                res = fn(df, cfg)
            except TypeError:
                res = fn(df)
            if isinstance(res, tuple):
                score, direction = res[0], res[1]
            else:
                score, direction = res, "none"
            symbol = ""
            if isinstance(cfg, dict):
                symbol = cfg.get("symbol", "")
            notifier.notify(
                f"\U0001F4C8 Signal: {symbol} \u2192 {direction.upper()} | Confidence: {score:.2f}"
            )
            return score, direction

        wrapped.__name__ = fn.__name__
        return wrapped

    if isinstance(regime, dict):
        if regime.get("1m") == "breakout" and regime.get("15m") == "trending":
            regime = "breakout"
        else:
            base = None
            if config:
                base = config.get("timeframe")
            regime = regime.get(base, next(iter(regime.values())))

    if mode == "onchain":
        if regime in {"breakout", "volatile"}:
            logger.info("Routing to sniper bot (onchain)")
            return _wrap(sniper_bot.generate_signal)
        logger.info("Routing to DEX scalper (onchain)")
        return _wrap(dex_scalper.generate_signal)

    if config and config.get("rl_selector", {}).get("enabled"):
        from .rl import strategy_selector as rl_selector

        strategy_fn = rl_selector.select_strategy(regime)
        logger.info("RL selector chose %s for %s", strategy_fn.__name__, regime)
        return _wrap(strategy_fn)

    if config and config.get("meta_selector", {}).get("enabled"):
        from . import meta_selector

        strategy_fn = meta_selector.choose_best(regime)
        logger.info("Meta selector chose %s for %s", strategy_fn.__name__, regime)
        return _wrap(strategy_fn)

    if config and config.get("signal_fusion", {}).get("enabled"):
        from . import meta_selector
        from crypto_bot.signals.signal_fusion import SignalFusionEngine

        pairs_conf = config.get("signal_fusion", {}).get("strategies", [])
        mapping = getattr(meta_selector, "_STRATEGY_FN_MAP", {})
        strategies: list[tuple[Callable[[pd.DataFrame], Tuple[float, str]], float]] = []
        for name, weight in pairs_conf:
            fn = mapping.get(name)
            if fn:
                strategies.append((fn, float(weight)))
        if not strategies:
            strategies.append((strategy_for(regime, config), 1.0))
        engine = SignalFusionEngine(strategies)

        def fused(df: pd.DataFrame, cfg=None):
            return engine.fuse(df, cfg)

        logger.info("Routing to signal fusion engine")
        return _wrap(fused)

    strategy_fn = strategy_for(regime, config)
    logger.info("Routing to %s (%s)", strategy_fn.__name__, mode)

    return _wrap(strategy_fn)
