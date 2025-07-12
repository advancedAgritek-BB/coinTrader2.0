from typing import Callable, Tuple, Dict, Iterable, Union, Mapping, Any

from dataclasses import dataclass, field, asdict

import pandas as pd
import numpy as np

from pathlib import Path
import yaml
import json
import time
from functools import lru_cache
from datetime import datetime

from crypto_bot.utils import timeframe_seconds, commit_lock
from crypto_bot.utils.strategy_utils import compute_edge

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.utils.cache_helpers import cache_by_id
from crypto_bot.selector import bandit

from crypto_bot.strategy import (
    trend_bot,
    grid_bot,
    sniper_bot,
    sniper_solana,
    dex_scalper,
    mean_bot,
    breakout_bot,
    micro_scalp_bot,
    bounce_scalper,
    cross_arbitrage,
    cross_chain_arbitrage,
)

logger = setup_logger(__name__, LOG_DIR / "bot.log")

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
    bandit_enabled: bool = False
    timeframe: str = "1h"
    timeframe_minutes: int = 60
    commit_lock_intervals: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RouterConfig":
        """Create ``RouterConfig`` from a dictionary (e.g. YAML)."""
        router = data.get("strategy_router", {})
        fusion = data.get("signal_fusion", {})
        tf = str(data.get("timeframe", "1h"))
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
            bandit_enabled=bool(data.get("bandit", {}).get("enabled", False)),
            timeframe=tf,
            timeframe_minutes=int(pd.Timedelta(tf).total_seconds() // 60),
            commit_lock_intervals=int(router.get("commit_lock_intervals", 0)),
            raw=data,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the underlying raw dictionary."""
        if isinstance(self.raw, Mapping):
            return dict(self.raw)
        return asdict(self)


DEFAULT_ROUTER_CFG = RouterConfig.from_dict(DEFAULT_CONFIG)


def cfg_get(cfg: RouterConfig | Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Return configuration value ``key`` from ``cfg``.

    The helper accepts either a :class:`RouterConfig` or a mapping based
    configuration dictionary.
    """
    if isinstance(cfg, RouterConfig):
        return getattr(cfg, key, default)
    if isinstance(cfg, Mapping):
        router = cfg.get("strategy_router", {})
        if key in router:
            return router.get(key, default)
        return cfg.get(key, default)
    return default

def cfg_get(cfg: Mapping[str, Any] | RouterConfig, key: str, default: Any | None = None) -> Any:
    """Return a configuration value for ``key`` from ``cfg``.

    When ``cfg`` is a :class:`RouterConfig` the lookup is performed on
    ``cfg.raw``. Keys missing at the top level are looked up under the
    ``"strategy_router"`` section.  Otherwise ``dict.get`` is used on ``cfg``
    directly.
    """
    if isinstance(cfg, RouterConfig):
        if isinstance(cfg.raw, Mapping):
            if key in cfg.raw:
                return cfg.raw.get(key, default)
            return cfg.raw.get("strategy_router", {}).get(key, default)
        return default
    return cfg.get(key, default)


# Path storing the last selected regime and timestamp
LAST_REGIME_FILE = LOG_DIR / "last_regime.json"


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


def get_strategy_by_name(
    name: str,
) -> Callable[[pd.DataFrame], Tuple[float, str]] | None:
    """Return strategy callable for ``name`` if available."""
    from . import meta_selector
    from .rl import strategy_selector as rl_selector

    mapping: Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]] = {}
    mapping.update(getattr(meta_selector, "_STRATEGY_FN_MAP", {}))
    mapping.update(getattr(rl_selector, "_STRATEGY_FN_MAP", {}))
    mapping.setdefault("cross_arbitrage", cross_arbitrage.generate_signal)
    mapping.setdefault(
        "cross_chain_arbitrage", cross_chain_arbitrage.generate_signal
    )
    return mapping.get(name)


@cache_by_id
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


_CONFIG_REGISTRY: Dict[int, Mapping[str, Any] | RouterConfig] = {}


def _register_config(cfg: Mapping[str, Any] | RouterConfig) -> int:
    """Register config and return its id for cache lookups."""
    cid = id(cfg)
    _CONFIG_REGISTRY[cid] = cfg
    return cid


@lru_cache(maxsize=8)
def _build_mappings_cached(config_id: int) -> tuple[
    Dict[str, Callable[[pd.DataFrame], Tuple[float, str]]],
    Dict[str, list[Callable[[pd.DataFrame], Tuple[float, str]]]],
]:
    cfg = _CONFIG_REGISTRY.get(config_id, DEFAULT_ROUTER_CFG)
    return _build_mappings(cfg)

_register_config(DEFAULT_ROUTER_CFG)
STRATEGY_MAP, REGIME_STRATEGIES = _build_mappings_cached(id(DEFAULT_ROUTER_CFG))


def strategy_for(
    regime: str, config: RouterConfig | Mapping[str, Any] | None = None
) -> Callable[[pd.DataFrame], Tuple[float, str]]:
    """Return strategy callable for a given regime."""
    cfg = config or DEFAULT_ROUTER_CFG
    _register_config(cfg)
    mapping, _ = _build_mappings_cached(id(cfg))
    return mapping.get(regime, grid_bot.generate_signal)


def get_strategies_for_regime(
    regime: str, config: RouterConfig | Mapping[str, Any] | None = None
) -> list[Callable[[pd.DataFrame], Tuple[float, str]]]:
    """Return list of strategies mapped to ``regime``."""
    cfg = config or DEFAULT_ROUTER_CFG
    _register_config(cfg)
    _, mapping = _build_mappings_cached(id(cfg))
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


def _bandit_context(df: pd.DataFrame, regime: str) -> Dict[str, float]:
    """Return bandit context features for Thompson sampling."""
    context: Dict[str, float] = {}
    for r in [
        "trending",
        "sideways",
        "mean-reverting",
        "breakout",
        "volatile",
        "unknown",
    ]:
        context[f"regime_{r}"] = 1.0 if regime == r else 0.0

    try:
        from crypto_bot.volatility_filter import calc_atr
        from crypto_bot.utils import stats
    except Exception:
        return context

    if df is not None and not df.empty:
        price = df["close"].iloc[-1]
        atr = calc_atr(df)
        context["atr_pct"] = atr / price if price else 0.0

        ts = df.index[-1]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.to_datetime(ts)
        hour = ts.hour + ts.minute / 60
        context["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
        context["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
        vol_z = stats.zscore(df["volume"], lookback=20)
        context["liquidity_z"] = float(vol_z.iloc[-1]) if not vol_z.empty else 0.0
    return context


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
    df: pd.DataFrame | None = None,
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

    df : pd.DataFrame | None
        Optional dataframe used for fast-path checks. When provided the router
        may immediately return a strategy without additional context.

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
                f"\U0001f4c8 Signal: {symbol} \u2192 {direction.upper()} | Confidence: {score:.2f}"
            )
            return score, direction

        wrapped.__name__ = fn.__name__
        return wrapped

    cfg = config or DEFAULT_ROUTER_CFG

    cca_cfg = cfg_get(cfg, "cross_chain_arbitrage", {})
    if isinstance(cca_cfg, Mapping) and cca_cfg.get("enabled"):
        exchanges = cca_cfg.get("exchanges") or cca_cfg.get("exchange_list")
        symbol = cca_cfg.get("symbol") or cfg_get(cfg, "symbol", "")
        threshold = float(cca_cfg.get("threshold", 0.005))
        if exchanges:
            logger.info("Routing to cross_chain_arbitrage strategy")

            def cca_fn(_df: pd.DataFrame | None = None, _cfg=None):
                return cross_chain_arbitrage.generate_signal(
                    exchanges, symbol, threshold
                )

            return _wrap(cca_fn)

    ca_cfg = cfg_get(cfg, "cross_arbitrage", {})
    if isinstance(ca_cfg, Mapping) and ca_cfg.get("enabled"):
        ex_a = ca_cfg.get("exchange_a")
        ex_b = ca_cfg.get("exchange_b")
        symbol = ca_cfg.get("symbol") or cfg_get(cfg, "symbol", "")
        threshold = float(ca_cfg.get("threshold", 0.005))
        if ex_a and ex_b:
            logger.info("Routing to cross_arbitrage strategy")

            def cross_fn(_df: pd.DataFrame | None = None, _cfg=None):
                return cross_arbitrage.generate_signal(ex_a, ex_b, symbol, threshold)

            return _wrap(cross_fn)

    # === FAST-PATH FOR STRONG SIGNALS ===
    fp = (
        cfg.raw.get("strategy_router", {}).get("fast_path", {})
        if hasattr(cfg, "raw")
        else cfg.get("strategy_router", {}).get("fast_path", {})
    )
    if df is not None:
        try:
            # 1) breakout squeeze detected by Bollinger band z-score and
            #    concurrent volume spike
            from ta.volatility import BollingerBands

            window = int(fp.get("breakout_squeeze_window", 20))
            bw_z_thr = float(fp.get("breakout_bandwidth_zscore", -0.84))
            vol_mult = float(fp.get("breakout_volume_multiplier", 5))
            max_bw = float(fp.get("breakout_max_bandwidth", 0.05))

            bb = BollingerBands(df["close"], window=window)
            wband_series = bb.bollinger_wband()
            wband = wband_series.iloc[-1]
            w_mean = wband_series.rolling(window).mean().iloc[-1]
            w_std = wband_series.rolling(window).std().iloc[-1]
            z = (wband - w_mean) / w_std if w_std > 0 else float("inf")
            vol_mean = df["volume"].rolling(window).mean().iloc[-1]
            if z < bw_z_thr and df["volume"].iloc[-1] > vol_mean * vol_mult:
                logger.info(
                    "FAST-PATH: breakout_bot via bandwidth z-score and volume spike"
                )
                return _wrap(breakout_bot.generate_signal)
            z_series = (
                wband_series - wband_series.rolling(window).mean()
            ) / wband_series.rolling(window).std()
            vol_ma = df["volume"].rolling(window).mean()

            if (
                z_series.iloc[-1] < -0.84
                and wband < max_bw
                and df["volume"].iloc[-1] > vol_ma.iloc[-1] * vol_mult
            ):
                logger.info(
                    "FAST-PATH: breakout_bot via BB squeeze z-score + volume spike"
                )
                return _wrap(breakout_bot.generate_signal)

            # 2) ultra-strong trend by ADX
            from ta.trend import ADXIndicator

            adx_thr = float(fp.get("trend_adx_threshold", 35))
            adx_val = (
                ADXIndicator(df["high"], df["low"], df["close"], window=window)
                .adx()
                .iloc[-1]
            )
            if adx_val > adx_thr:
                logger.info("FAST-PATH: trend_bot via ADX > %.1f", adx_thr)
                return _wrap(trend_bot.generate_signal)
        except Exception:  # pragma: no cover - safety
            pass
    # === end fast-path ===

    if isinstance(regime, dict):
        if regime.get("1m") == "breakout" and regime.get("15m") == "trending":
            regime = "breakout"
        else:
            base = (
                cfg.timeframe if isinstance(cfg, RouterConfig) else cfg.get("timeframe")
            )
            regime = regime.get(base, next(iter(regime.values())))

    tf_sec = timeframe_seconds(
        None,
        cfg.timeframe if isinstance(cfg, RouterConfig) else cfg.get("timeframe", "1h"),
    )
    regime = commit_lock.check_and_update(
        regime,
        tf_sec,
        cfg_get(cfg, "commit_lock_intervals", 0),
    )



    # commit lock logic
    intervals = int(cfg_get(cfg, "commit_lock_intervals", 0))
    if intervals:
        lock_file = Path("last_regime.json")
        last_reg = None
        last_ts = 0.0
        if lock_file.exists():
            try:
                data = json.loads(lock_file.read_text())
                last_reg = data.get("regime")
                last_ts = float(data.get("timestamp", 0))
            except Exception:
                pass

        tf = cfg_get(cfg, "timeframe", "1h")
        interval = timeframe_seconds(None, tf)
        now = time.time()
        if last_reg and regime != last_reg and now - last_ts < interval * intervals:
            regime = last_reg
        else:
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            lock_file.write_text(json.dumps({"regime": regime, "timestamp": now}))

    tf = cfg_get(cfg, "timeframe", "1h")
    tf_minutes = (
        cfg.timeframe_minutes
        if isinstance(cfg, RouterConfig)
        else getattr(cfg, "timeframe_minutes", int(pd.Timedelta(tf).total_seconds() // 60))
    )

    LAST_REGIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    last_data = {}
    if LAST_REGIME_FILE.exists():
        try:
            last_data = json.loads(LAST_REGIME_FILE.read_text())
        except Exception:
            last_data = {}
    last_ts = last_data.get("timestamp")
    last_regime = last_data.get("regime")
    if last_ts and last_regime:
        try:
            ts = datetime.fromisoformat(last_ts)
            if (datetime.utcnow() - ts).total_seconds() < tf_minutes * 60 * 3:
                regime = last_regime
        except Exception:
            pass
    LAST_REGIME_FILE.write_text(
        json.dumps({"timestamp": datetime.utcnow().isoformat(), "regime": regime})
    )

    # Thompson sampling router
    bandit_active = (
        cfg.bandit_enabled
        if isinstance(cfg, RouterConfig)
        else bool(cfg.get("bandit", {}).get("enabled"))
    )
    if bandit_active:
        strategies = get_strategies_for_regime(regime, cfg)
        if isinstance(cfg, RouterConfig):
            arms = list(cfg.regimes.get(regime, []))
        else:
            arms = list(
                cfg.get("strategy_router", {}).get("regimes", {}).get(regime, [])
            )
        arms = [a for a in arms if get_strategy_by_name(a)]
        if not arms:
            arms = [fn.__name__ for fn in strategies]
        symbol = ""
        if isinstance(cfg, RouterConfig):
            symbol = str(cfg.raw.get("symbol", ""))
        elif isinstance(cfg, Mapping):
            symbol = str(cfg.get("symbol", ""))
        context_df = df if df is not None else pd.DataFrame()
        context = _bandit_context(context_df, regime)
        choice = bandit.select(context, arms, symbol)
        fn = get_strategy_by_name(choice)
        if fn:
            logger.info("Bandit selected %s for %s", choice, regime)
            return _wrap(fn)

    edge_thr = float(cfg_get(cfg, "scalp_edge_threshold", 0.0))
    if edge_thr:
        penalty = float(cfg_get(cfg, "drawdown_penalty_coef", 0.0))
        try:
            micro = compute_edge("micro_scalp_bot", penalty)
            bounce = compute_edge("bounce_scalper", penalty)
            best_edge = max(micro, bounce)
            if best_edge > edge_thr:
                best_fn = micro_scalp_bot if micro >= bounce else bounce_scalper
                logger.info(
                    "Routing to %s via edge %.2f > %.2f",
                    best_fn.__name__,
                    best_edge,
                    edge_thr,
                )
                return _wrap(best_fn.generate_signal)
        except Exception:
            pass

    if mode == "onchain":
        chain = ""
        if isinstance(cfg, RouterConfig):
            chain = str(cfg.raw.get("chain") or cfg.raw.get("preferred_chain", ""))
        elif isinstance(cfg, Mapping):
            chain = str(cfg.get("chain") or cfg.get("preferred_chain", ""))

        # Route Solana on-chain mode directly to the Solana sniper bot
        if chain.lower().startswith("sol"):
            logger.info("Routing to Solana sniper bot (onchain)")
            return _wrap(sniper_solana.generate_signal)

        if regime in {"breakout", "volatile"}:
            logger.info("Routing to sniper bot (onchain)")
            return _wrap(sniper_bot.generate_signal)
        logger.info("Routing to DEX scalper (onchain)")
        return _wrap(dex_scalper.generate_signal)

    select_df = df if df is not None else pd.DataFrame()
    strategy_fn = Selector(cfg).select(select_df, regime, mode, notifier)
    return _wrap(strategy_fn)
