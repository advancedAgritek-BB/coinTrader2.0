from typing import Callable, Tuple, Dict, Iterable, Union, Mapping, Any

import asyncio

from dataclasses import dataclass, field, asdict
import redis

import pandas as pd
import numpy as np

from pathlib import Path
import yaml
import json
import time
from functools import lru_cache
from datetime import datetime

from crypto_bot.utils import timeframe_seconds, commit_lock

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.telemetry import telemetry
import threading
from collections import defaultdict
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.utils.cache_helpers import cache_by_id
from crypto_bot.utils.token_registry import TOKEN_MINTS
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
)

logger = setup_logger(__name__, LOG_DIR / "bot.log")
_SYMBOL_LOCKS: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CONFIG_PATH) as f:
    DEFAULT_CONFIG = yaml.safe_load(f)

# Map symbols to asyncio locks guarding order placement
symbol_locks: Dict[str, asyncio.Lock] = {}

# Event loop captured when locks are first acquired
_LOCK_LOOP: asyncio.AbstractEventLoop | None = None


async def acquire_symbol_lock(symbol: str) -> None:
    """Acquire the asyncio lock associated with ``symbol``."""
    global _LOCK_LOOP
    if _LOCK_LOOP is None:
        _LOCK_LOOP = asyncio.get_running_loop()
    lock = symbol_locks.setdefault(symbol, asyncio.Lock())
    await lock.acquire()


async def release_symbol_lock(symbol: str) -> None:
    """Release the lock for ``symbol`` if held."""
    lock = symbol_locks.get(symbol)
    if lock and lock.locked():
        lock.release()


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
    trending_timeframe: str | None = None
    volatile_timeframe: str | None = None
    sideways_timeframe: str | None = None
    scalp_timeframe: str | None = None
    breakout_timeframe: str | None = None
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
            trending_timeframe=str(router.get("trending_timeframe", data.get("trending_timeframe", tf))) or None,
            volatile_timeframe=str(router.get("volatile_timeframe", data.get("volatile_timeframe", tf))) or None,
            sideways_timeframe=str(router.get("sideways_timeframe", data.get("sideways_timeframe", tf))) or None,
            scalp_timeframe=str(router.get("scalp_timeframe", data.get("scalp_timeframe", tf))) or None,
            breakout_timeframe=str(router.get("breakout_timeframe", data.get("breakout_timeframe", tf))) or None,
            commit_lock_intervals=int(router.get("commit_lock_intervals", 0)),
            raw=data,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return the underlying raw dictionary."""
        if isinstance(self.raw, Mapping):
            return dict(self.raw)
        return asdict(self)


DEFAULT_ROUTER_CFG = RouterConfig.from_dict(DEFAULT_CONFIG)


@dataclass
class BotStats:
    """Performance statistics for a trading bot."""

    sharpe_30d: float = 0.0
    win_rate_30d: float = 0.0
    avg_r_multiple: float = 0.0


def load_bot_stats(name: str) -> BotStats:
    """Return statistics for ``name`` loaded from Redis."""
    try:
        r = redis.Redis()
        raw = r.get(f"bot-stats:{name}")
    except Exception:
        return BotStats()
    if not raw:
        return BotStats()
    try:
        if isinstance(raw, bytes):
            raw = raw.decode()
        data = json.loads(raw)
    except Exception:
        return BotStats()
    return BotStats(
        sharpe_30d=float(data.get("sharpe_30d", 0.0)),
        win_rate_30d=float(data.get("win_rate_30d", 0.0)),
        avg_r_multiple=float(data.get("avg_r_multiple", 0.0)),
    )


def score_bot(stats: BotStats) -> float:
    """Return a ranking score for ``stats``."""
    return (
        stats.sharpe_30d * 0.4
        + stats.win_rate_30d * 0.3
        + stats.avg_r_multiple * 0.3
    )


def cfg_get(cfg: Mapping[str, Any] | RouterConfig, key: str, default: Any | None = None) -> Any:
    """Return configuration value ``key`` from ``cfg``.

    Supports both :class:`RouterConfig` instances and plain mapping objects. For
    ``RouterConfig`` the dataclass attributes are checked first, then the
    underlying ``raw`` mapping including the ``"strategy_router"`` section. For
    mappings the lookup is performed on the top level and falls back to the
    ``"strategy_router"`` subsection.
    """
    if isinstance(cfg, RouterConfig):
        if hasattr(cfg, key):
            return getattr(cfg, key, default)
        if isinstance(cfg.raw, Mapping):
            if key in cfg.raw:
                return cfg.raw.get(key, default)
            return cfg.raw.get("strategy_router", {}).get(key, default)
        return default
    if isinstance(cfg, Mapping):
        if key in cfg:
            return cfg.get(key, default)
        return cfg.get("strategy_router", {}).get(key, default)
    return default


def wrap_with_tf(fn: Callable[[pd.DataFrame], Tuple[float, str]], tf: str):
    """Return ``fn`` wrapped to extract ``tf`` from a dataframe map."""

    def wrapped(df_or_map: Any, cfg=None):
        df = None
        if isinstance(df_or_map, Mapping):
            df = df_or_map.get(tf)
        if df is None:
            df = df_or_map if not isinstance(df_or_map, Mapping) else pd.DataFrame()
        try:
            return fn(df, cfg)
        except TypeError:
            return fn(df)

    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapped


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
    strategies = get_strategies_for_regime(regime, cfg)
    if strategies:
        base = strategies[0]
    else:
        base = breakout_bot.generate_signal if regime == "unknown" else grid_bot.generate_signal
    tf_key = f"{regime.replace('-', '_')}_timeframe"
    tf = cfg_get(cfg, tf_key, cfg_get(cfg, "timeframe", "1h"))
    return wrap_with_tf(base, tf)


def get_strategies_for_regime(
    regime: str, config: RouterConfig | Mapping[str, Any] | None = None
) -> list[Callable[[pd.DataFrame], Tuple[float, str]]]:
    """Return list of strategies mapped to ``regime``."""
    cfg = config or DEFAULT_ROUTER_CFG
    _register_config(cfg)
    if isinstance(cfg, RouterConfig):
        names = cfg.regimes.get(regime, [])
    else:
        names = cfg.get("strategy_router", {}).get("regimes", {}).get(regime, [])
    if isinstance(names, str):
        names = [names]
    pairs: list[tuple[str, Callable[[pd.DataFrame], Tuple[float, str]]]] = []
    for name in names:
        fn = get_strategy_by_name(name)
        if fn:
            pairs.append((name, fn))
    if not pairs:
        _, mapping = _build_mappings_cached(id(cfg))
        return mapping.get(regime, [grid_bot.generate_signal])
    pairs.sort(key=lambda p: score_bot(load_bot_stats(p[0])), reverse=True)
    return [fn for _, fn in pairs]


def evaluate_regime(
    regime: str,
    df: pd.DataFrame,
    config: RouterConfig | Mapping[str, Any] | None = None,
) -> Tuple[float, str]:
    """Evaluate and fuse all strategies assigned to ``regime``."""
    cfg = config or DEFAULT_ROUTER_CFG
    tf_key = f"{regime.replace('-', '_')}_timeframe"
    tf = cfg_get(cfg, tf_key, cfg_get(cfg, "timeframe", "1h"))
    strategies = [wrap_with_tf(s, tf) for s in get_strategies_for_regime(regime, cfg)]

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

    def _instrument(fn: Callable[[pd.DataFrame], Tuple[float, str]]):
        def wrapped(df_p: pd.DataFrame, cfg_p=None):
            telemetry.inc("router.signals_checked")
            try:
                res = fn(df_p, cfg_p)
            except TypeError:
                res = fn(df_p)
            except asyncio.TimeoutError:
                telemetry.inc("router.signal_timeout")
                raise
            telemetry.inc("router.signal_returned")
            return res

        wrapped.__name__ = getattr(fn, "__name__", "wrapped")
        return wrapped

    for fn in strategies:
        w = float(weights.get(fn.__name__, 1.0))
        if w < min_conf:
            continue
        pairs.append((_instrument(fn), w))

    if not pairs:
        pairs.append((strategies[0], 1.0))

    from crypto_bot.signals.signal_fusion import SignalFusionEngine

    engine = SignalFusionEngine(pairs)
    return engine.fuse(df, cfg_dict)


def _bandit_context(
    df: pd.DataFrame, regime: str, symbol: str | None = None
) -> Dict[str, float]:
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
        if symbol:
            try:
                from crypto_bot.utils.pyth import get_pyth_price

                pyth_price = get_pyth_price(symbol)
                if pyth_price:
                    price = pyth_price
            except Exception:
                pass

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
    df_map: Mapping[str, pd.DataFrame] | pd.DataFrame | None = None,
) -> Callable[[pd.DataFrame | Mapping[str, pd.DataFrame]], Tuple[float, str]]:
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

    df_map : Mapping[str, pd.DataFrame] | pd.DataFrame | None
        Optional dataframe or mapping used for fast-path checks. When provided
        the router may immediately return a strategy without additional
        context.

    Returns
    -------
    Callable[[pd.DataFrame | Mapping[str, pd.DataFrame]], Tuple[float, str]]
        Strategy function returning a score and trade direction.
    """

    def _wrap(fn: Callable[[pd.DataFrame], Tuple[float, str]]):
        async def inner(df: pd.DataFrame, cfg=None):
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
            if direction != "none" and symbol:
                await acquire_symbol_lock(symbol)
            if notifier is not None:
                notifier.notify(
                    f"\U0001f4c8 Signal: {symbol} \u2192 {direction.upper()} | Confidence: {score:.2f}"
                )
            return score, direction

        def wrapped(df: pd.DataFrame, cfg=None):
            coro = inner(df, cfg)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)
            else:
                return coro

        wrapped.__name__ = fn.__name__
        return wrapped

    cfg = config or DEFAULT_ROUTER_CFG
    df = None
    if isinstance(df_map, pd.DataFrame):
        df = df_map
    elif isinstance(df_map, Mapping):
        df = next(iter(df_map.values()), None)

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

            window = int(fp.get("breakout_squeeze_window", 15))
            bw_z_thr = float(fp.get("breakout_bandwidth_zscore", -0.84))
            vol_mult = float(fp.get("breakout_volume_multiplier", 4))
            max_bw = float(fp.get("breakout_max_bandwidth", 0.04))

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

            adx_thr = float(fp.get("trend_adx_threshold", 25))
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

    symbol = ""
    chain = ""
    if isinstance(cfg, RouterConfig):
        symbol = str(cfg.raw.get("symbol", ""))
        chain = str(cfg.raw.get("chain") or cfg.raw.get("preferred_chain", ""))
        grid_cfg = cfg.raw.get("grid_bot", {})
    elif isinstance(cfg, Mapping):
        symbol = str(cfg.get("symbol", ""))
        chain = str(cfg.get("chain") or cfg.get("preferred_chain", ""))
        grid_cfg = cfg.get("grid_bot", {})
    else:
        grid_cfg = {}
    if symbol.endswith("/USDC") and mode == "auto":
        base = symbol.split("/")[0]
        if base.upper() in TOKEN_MINTS:
            logger.info("Routing %s pair to Solana sniper bot (auto)", symbol)
            return _wrap(sniper_solana.generate_signal)
        logger.info("Mint for %s not found; falling back to CEX", base.upper())
        select_df = df if df is not None else pd.DataFrame()
        strategy_fn = Selector(cfg).select(select_df, regime, "cex", notifier)
        return _wrap(strategy_fn)
    if symbol.endswith("/USDC") and regime == "breakout":
        base = symbol.split("/")[0]
        if base.upper() in TOKEN_MINTS:
            logger.info("Routing USDC breakout to Solana sniper bot")
            return _wrap(sniper_solana.generate_signal)

    if chain.lower().startswith("sol") and mode in {"auto", "onchain"} and regime in {"breakout", "volatile"}:
        base = symbol.split("/")[0] if symbol else ""
        if not symbol or base.upper() in TOKEN_MINTS:
            logger.info("Routing %s regime to Solana sniper bot (%s mode)", regime, mode)
            return _wrap(sniper_solana.generate_signal)

    if regime == "sideways" and grid_cfg.get("dynamic_grid") and symbol:
        logger.info("Routing dynamic grid signal to micro scalp bot")
        return _wrap(micro_scalp_bot.generate_signal)

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
        context = _bandit_context(context_df, regime, symbol)
        choice = bandit.select(context, arms, symbol)
        fn = get_strategy_by_name(choice)
        if fn:
            logger.info("Bandit selected %s for %s", choice, regime)
            return _wrap(fn)

    if mode == "onchain":
        if chain.lower().startswith("sol"):
            if regime in {"breakout", "volatile"}:
                logger.info("Routing to Solana sniper bot (onchain)")
                return _wrap(sniper_solana.generate_signal)
            logger.info("Routing to DEX scalper (onchain)")
            return _wrap(dex_scalper.generate_signal)

        if regime in {"breakout", "volatile"}:
            logger.info("Routing to sniper bot (onchain)")
            return _wrap(sniper_bot.generate_signal)
        logger.info("Routing to DEX scalper (onchain)")
        return _wrap(dex_scalper.generate_signal)

    select_df = df if df is not None else pd.DataFrame()
    strategy_fn = Selector(cfg).select(select_df, regime, mode, notifier)
    return _wrap(strategy_fn)
