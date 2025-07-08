from typing import Callable, Tuple, Dict, Iterable, Union, Mapping, Any

from dataclasses import dataclass, field, asdict

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


@dataclass
class RouterConfig:
    """Configuration for routing strategies."""

    regimes: Dict[str, Iterable[str]] = field(default_factory=dict)
    min_score: float = 0.0
    fusion_method: str = "weight"
    perf_window: int = 20
    min_confidence: float = 0.0
    fusion_enabled: bool = False
    strategies: list[tuple[str, float]] = field(default_factory=list)
    rl_selector: bool = False
    meta_selector: bool = False
    timeframe: str = "1h"
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouterConfig":
        """Create ``RouterConfig`` from a dictionary (e.g. YAML)."""
        router = data.get("strategy_router", {})
        fusion = data.get("signal_fusion", {})
        return cls(
            regimes=router.get("regimes", {}),
            min_score=float(
                data.get("min_confidence_score", data.get("signal_threshold", 0.0))
            ),
            fusion_method=fusion.get("fusion_method", "weight"),
            perf_window=int(fusion.get("perf_window", 20)),
            min_confidence=float(fusion.get("min_confidence", 0.0)),
            fusion_enabled=bool(fusion.get("enabled", False)),
            strategies=[tuple(x) for x in fusion.get("strategies", [])],
            rl_selector=bool(data.get("rl_selector", {}).get("enabled", False)),
            meta_selector=bool(data.get("meta_selector", {}).get("enabled", False)),
            timeframe=str(data.get("timeframe", "1h")),
            raw=data,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the underlying raw dictionary."""
        if isinstance(self.raw, Mapping):
            return dict(self.raw)
        return asdict(self)


DEFAULT_ROUTER_CFG = RouterConfig.from_dict(DEFAULT_CONFIG)


class Selector:
    """Helper class to select a strategy callable."""

    def __init__(self, config: RouterConfig):
        self.config = config

    def select(
        self,
        df: pd.DataFrame,
        regime: str,
        mode: str,
        notifier=None,
    ) -> Callable[[pd.DataFrame], Tuple[float, str]]:
        cfg = self.config

        if (isinstance(cfg, RouterConfig) and cfg.rl_selector) or (
            not isinstance(cfg, RouterConfig)
            and cfg.get("rl_selector", {}).get("enabled")
        ):
            from .rl import strategy_selector as rl_selector

            strategy_fn = rl_selector.select_strategy(regime)
            logger.info("RL selector chose %s for %s", strategy_fn.__name__, regime)
            return strategy_fn

        if (isinstance(cfg, RouterConfig) and cfg.meta_selector) or (
            not isinstance(cfg, RouterConfig)
            and cfg.get("meta_selector", {}).get("enabled")
        ):
            from . import meta_selector

            strategy_fn = meta_selector.choose_best(regime)
            logger.info("Meta selector chose %s for %s", strategy_fn.__name__, regime)
            return strategy_fn

        if (isinstance(cfg, RouterConfig) and cfg.fusion_enabled) or (
            not isinstance(cfg, RouterConfig)
            and cfg.get("signal_fusion", {}).get("enabled")
        ):
            from . import meta_selector
            from crypto_bot.signals.signal_fusion import SignalFusionEngine

            pairs_conf = (
                cfg.strategies
                if isinstance(cfg, RouterConfig)
                else cfg.get("signal_fusion", {}).get("strategies", [])
            )
            mapping = getattr(meta_selector, "_STRATEGY_FN_MAP", {})
            strategies: list[
                tuple[Callable[[pd.DataFrame], Tuple[float, str]], float]
            ] = []
            for name, weight in pairs_conf:
                fn = mapping.get(name)
                if fn:
                    strategies.append((fn, float(weight)))
            if not strategies:
                strategies.append((strategy_for(regime, cfg), 1.0))
            engine = SignalFusionEngine(strategies)

            def fused(df: pd.DataFrame, cfg_param=None):
                return engine.fuse(
                    df,
                    cfg.as_dict() if isinstance(cfg, RouterConfig) else cfg_param,
                )

            logger.info("Routing to signal fusion engine")
            return fused

        strategy_fn = strategy_for(regime, cfg)
        logger.info("Routing to %s (%s)", strategy_fn.__name__, mode)
        return strategy_fn

def get_strategy_by_name(name: str) -> Callable[[pd.DataFrame], Tuple[float, str]] | None:
    """Return strategy callable for ``name`` if available."""
    from . import meta_selector
    from .rl import strategy_selector as rl_selector

    mapping: Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]] = {}
    mapping.update(getattr(meta_selector, "_STRATEGY_FN_MAP", {}))
    mapping.update(getattr(rl_selector, "_STRATEGY_FN_MAP", {}))
    return mapping.get(name)


def _build_mappings(config: Mapping[str, Any] | RouterConfig) -> tuple[
    Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]],
    Dict[str, list[Callable[[pd.DataFrame], Tuple[float, str]]]],
]:
    """Return mapping dictionaries from configuration."""
    if isinstance(config, RouterConfig):
        regimes = config.regimes
    else:
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


STRATEGY_MAP, REGIME_STRATEGIES = _build_mappings(DEFAULT_ROUTER_CFG)


def strategy_for(regime: str, config: RouterConfig | Mapping[str, Any] | None = None) -> Callable[[pd.DataFrame], Tuple[float, str]]:
    """Return strategy callable for a given regime."""
    mapping, _ = _build_mappings(config or DEFAULT_ROUTER_CFG)
    return mapping.get(regime, grid_bot.generate_signal)


def get_strategies_for_regime(regime: str, config: RouterConfig | Mapping[str, Any] | None = None) -> list[Callable[[pd.DataFrame], Tuple[float, str]]]:
    """Return list of strategies mapped to ``regime``."""
    _, mapping = _build_mappings(config or DEFAULT_ROUTER_CFG)
    return mapping.get(regime, [grid_bot.generate_signal])


def evaluate_regime(
    regime: str,
    df: pd.DataFrame,
    config: RouterConfig | Mapping[str, Any] | None = None,
) -> Tuple[float, str]:
    """Evaluate and fuse all strategies assigned to ``regime``."""
    cfg = config or DEFAULT_ROUTER_CFG
    strategies = get_strategies_for_regime(regime, cfg)

    if isinstance(cfg, RouterConfig):
        method = cfg.fusion_method
        min_conf = cfg.min_confidence
        cfg_dict = cfg.as_dict()
    else:
        fusion_cfg = cfg.get("signal_fusion", {})
        method = fusion_cfg.get("fusion_method", "weight")
        min_conf = float(fusion_cfg.get("min_confidence", 0.0))
        cfg_dict = cfg

    weights = {}
    if method == "weight":
        from crypto_bot.utils.regime_pnl_tracker import compute_weights

        weights = compute_weights(regime)

    pairs: list[Tuple[Callable[[pd.DataFrame], Tuple[float, str]], float]] = []
    for fn in strategies:
        w = float(weights.get(fn.__name__, 1.0))
        if w < min_conf:
            continue
        pairs.append((fn, w))

    if not pairs:
        pairs.append((strategies[0], 1.0))

    from crypto_bot.signals.signal_fusion import SignalFusionEngine

    engine = SignalFusionEngine(pairs)
    return engine.fuse(df, cfg_dict)


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
    config: RouterConfig | Mapping[str, Any] | None = None,
    notifier: TelegramNotifier | None = None,
) -> Callable[[pd.DataFrame], Tuple[float, str]]:
    """Select a strategy based on market regime and operating mode.

    Parameters
    ----------
    regime : str
        Current market regime as classified by indicators.
    mode : str
        Trading environment, either ``cex``, ``onchain`` or ``auto``.
    config : RouterConfig | dict | None
        Optional configuration object. When ``meta_selector.enabled`` is
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

    cfg = config or DEFAULT_ROUTER_CFG

    if isinstance(regime, dict):
        if regime.get("1m") == "breakout" and regime.get("15m") == "trending":
            regime = "breakout"
        else:
            base = cfg.timeframe if isinstance(cfg, RouterConfig) else cfg.get("timeframe")
            regime = regime.get(base, next(iter(regime.values())))

    if mode == "onchain":
        if regime in {"breakout", "volatile"}:
            logger.info("Routing to sniper bot (onchain)")
            return _wrap(sniper_bot.generate_signal)
        logger.info("Routing to DEX scalper (onchain)")
        return _wrap(dex_scalper.generate_signal)

    strategy_fn = Selector(cfg).select(pd.DataFrame(), regime, mode, notifier)
    return _wrap(strategy_fn)
