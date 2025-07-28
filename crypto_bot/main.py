import os
import sys
import asyncio
import contextlib
import time
from pathlib import Path
from datetime import datetime
from collections import deque, OrderedDict
from dataclasses import dataclass, field
import inspect

import aiohttp

try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    import types

    ccxt = types.SimpleNamespace()

import pandas as pd
import numpy as np
import yaml
from dotenv import dotenv_values
from pydantic import ValidationError

from schema.scanner import (
    ScannerConfig,
    SolanaScannerConfig,
    PythConfig,
)

from crypto_bot.utils.telegram import TelegramNotifier, send_test_message
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.portfolio_rotator import PortfolioRotator
from crypto_bot.wallet_manager import load_or_create
from crypto_bot.utils.market_analyzer import analyze_symbol
from crypto_bot.strategy import dca_bot
from crypto_bot.cooldown_manager import (
    configure as cooldown_configure,
)
from crypto_bot.phase_runner import BotContext, PhaseRunner
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
from crypto_bot.risk.exit_manager import (
    calculate_trailing_stop,
    should_exit,
)
from crypto_bot.execution.cex_executor import (
    execute_trade_async as cex_trade_async,
    get_exchange,
    get_exchanges,
)
from crypto_bot.open_position_guard import OpenPositionGuard
from crypto_bot import console_monitor, console_control
from crypto_bot.utils.position_logger import log_position, log_balance
from crypto_bot.utils.market_loader import (
    load_kraken_symbols,
    update_ohlcv_cache,
    update_multi_tf_ohlcv_cache,
    update_regime_tf_cache,
    timeframe_seconds,
    configure as market_loader_configure,
    fetch_order_book_async,
)
from crypto_bot.utils.pair_cache import PAIR_FILE, load_liquid_pairs
from tasks.refresh_pairs import (
    DEFAULT_MIN_VOLUME_USD,
    DEFAULT_TOP_K,
    refresh_pairs_async,
)
from crypto_bot.utils.eval_queue import build_priority_queue
from crypto_bot.solana import get_solana_new_tokens
from crypto_bot.utils.symbol_utils import get_filtered_symbols, fix_symbol
from crypto_bot.utils import symbol_utils
from crypto_bot.utils.metrics_logger import log_cycle as log_cycle_metrics
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.utils.strategy_utils import compute_strategy_weights
from crypto_bot.auto_optimizer import optimize_strategies
from crypto_bot.utils.telemetry import write_cycle_metrics
from crypto_bot.utils.token_registry import TOKEN_MINTS
from crypto_bot.utils import regime_pnl_tracker
from crypto_bot.utils import pnl_logger, regime_pnl_tracker

from crypto_bot.monitoring import record_sol_scanner_metrics
from crypto_bot.fund_manager import (
    auto_convert_funds,
    check_wallet_balances,
    detect_non_trade_tokens,
)
from crypto_bot.regime.regime_classifier import (
    classify_regime_async,
    classify_regime_cached,
)
from crypto_bot.volatility_filter import calc_atr
from crypto_bot.solana.exit import monitor_price
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor

# Backwards compatibility for tests
_fix_symbol = fix_symbol

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
ENV_PATH = Path(__file__).resolve().parent / ".env"

# Track the modification time of the loaded configuration
_LAST_CONFIG_MTIME = CONFIG_PATH.stat().st_mtime

logger = setup_logger("bot", LOG_DIR / "bot.log", to_console=False)

# Track WebSocket ping tasks
WS_PING_TASKS: set[asyncio.Task] = set()
# Track async sniper trade tasks
SNIPER_TASKS: set[asyncio.Task] = set()

# Queue of symbols awaiting evaluation across loops
symbol_priority_queue: deque[str] = deque()

# Protects shared queues for future multi-tasking scenarios
QUEUE_LOCK = asyncio.Lock()
# Protects OHLCV cache updates
OHLCV_LOCK = asyncio.Lock()

# Retry parameters for the initial symbol scan
MAX_SYMBOL_SCAN_ATTEMPTS = 3
SYMBOL_SCAN_RETRY_DELAY = 10
MAX_SYMBOL_SCAN_DELAY = 60

# Maximum number of symbols per timeframe to keep in the OHLCV cache
DF_CACHE_MAX_SIZE = 500

# Track regime analysis statistics
UNKNOWN_COUNT = 0
TOTAL_ANALYSES = 0


@dataclass
class SessionState:
    """Runtime session state shared across tasks."""

    positions: dict[str, dict] = field(default_factory=dict)
    df_cache: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    regime_cache: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    last_balance: float | None = None
    scan_task: asyncio.Task | None = None


def update_df_cache(
    cache: dict[str, dict[str, pd.DataFrame]],
    timeframe: str,
    symbol: str,
    df: pd.DataFrame,
    max_size: int = DF_CACHE_MAX_SIZE,
) -> None:
    """Update an OHLCV cache with LRU eviction."""
    tf_cache = cache.setdefault(timeframe, OrderedDict())
    if not isinstance(tf_cache, OrderedDict):
        tf_cache = OrderedDict(tf_cache)
        cache[timeframe] = tf_cache
    tf_cache[symbol] = df
    tf_cache.move_to_end(symbol)
    if len(tf_cache) > max_size:
        tf_cache.popitem(last=False)


def compute_average_atr(symbols: list[str], df_cache: dict, timeframe: str) -> float:
    """Return the average ATR for symbols present in ``df_cache``."""
    atr_values: list[float] = []
    tf_cache = df_cache.get(timeframe, {})
    for sym in symbols:
        df = tf_cache.get(sym)
        if df is None or df.empty or "close" not in df:
            continue
        atr_values.append(calc_atr(df))
    return sum(atr_values) / len(atr_values) if atr_values else 0.0


def is_market_pumping(
    symbols: list[str], df_cache: dict, timeframe: str = "1h", lookback_hours: int = 24
) -> bool:
    """Return ``True`` when the average % change over ``lookback_hours`` exceeds ~5%."""

    tf_cache = df_cache.get(timeframe, {})
    if not tf_cache:
        return False

    sec = timeframe_seconds(None, timeframe)
    candles = int(lookback_hours * 3600 / sec) if sec else 0
    changes: list[float] = []
    for sym in symbols:
        df = tf_cache.get(sym)
        if df is None or df.empty or "close" not in df:
            continue
        closes = df["close"]
        if len(closes) == 0:
            continue
        start_idx = -candles - 1 if candles and len(closes) > candles else 0
        try:
            start = float(closes[start_idx])
            end = float(closes[-1])
        except Exception:
            continue
        if start == 0:
            continue
        changes.append((end - start) / start)

    avg_change = sum(changes) / len(changes) if changes else 0.0
    return avg_change >= 0.05


async def get_market_regime(ctx: BotContext) -> str:
    """Return the market regime for the first cached symbol."""

    base_tf = ctx.config.get("timeframe", "1h")
    tf_cache = ctx.df_cache.get(base_tf, {})
    if not tf_cache:
        return "unknown"

    sym, df = next(iter(tf_cache.items()))
    higher_tf = ctx.config.get("higher_timeframe", "1d")
    higher_df = ctx.df_cache.get(higher_tf, {}).get(sym)
    label, info = await classify_regime_cached(sym, base_tf, df, higher_df)
    regime = label.split("_")[-1]
    logger.info(
        "Regime for %s [%s/%s]: %s",
        sym,
        base_tf,
        higher_tf,
        regime,
    )
    if regime == "unknown":
        logger.info(
            "Unknown regime details: bars=%s/%s info=%s",
            len(df) if df is not None else 0,
            len(higher_df) if isinstance(higher_df, pd.DataFrame) else 0,
            info,
        )
    return regime


def direction_to_side(direction: str) -> str:
    """Translate strategy direction to trade side."""
    return "buy" if direction == "long" else "sell"


def opposite_side(side: str) -> str:
    """Return the opposite trading side."""
    return "sell" if side == "buy" else "buy"


def _closest_wall_distance(book: dict, entry: float, side: str) -> float | None:
    """Return distance to the nearest bid/ask wall from ``entry``."""
    if not isinstance(book, dict):
        return None
    levels = book.get("asks") if side == "buy" else book.get("bids")
    if not levels:
        return None
    dists = []
    for price, _amount in levels:
        if side == "buy" and price > entry:
            dists.append(price - entry)
        elif side == "sell" and price < entry:
            dists.append(entry - price)
    if not dists:
        return None
    return min(dists)


def notify_balance_change(
    notifier: TelegramNotifier | None,
    previous: float | None,
    new_balance: float,
    enabled: bool,
) -> float:
    """Send a notification if the balance changed."""
    if notifier and enabled and previous is not None and new_balance != previous:
        notifier.notify(f"Balance changed: {new_balance:.2f} USDT")
    return new_balance


async def fetch_balance(exchange, paper_wallet, config):
    """Return the latest wallet balance without logging."""
    if config["execution_mode"] != "dry_run":
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
            bal = await exchange.fetch_balance()
        else:
            bal = await asyncio.to_thread(exchange.fetch_balance)
        return bal["USDT"]["free"] if isinstance(bal["USDT"], dict) else bal["USDT"]
    return paper_wallet.balance if paper_wallet else 0.0


async def fetch_and_log_balance(exchange, paper_wallet, config):
    """Return the latest wallet balance and log it."""
    latest_balance = await fetch_balance(exchange, paper_wallet, config)
    log_balance(float(latest_balance))
    return latest_balance


async def refresh_balance(ctx: BotContext) -> float:
    """Update ``ctx.balance`` from the exchange or paper wallet."""
    latest = await fetch_and_log_balance(
        ctx.exchange,
        ctx.paper_wallet,
        ctx.config,
    )
    ctx.balance = notify_balance_change(
        ctx.notifier,
        ctx.balance,
        float(latest),
        ctx.config.get("telegram", {}).get("balance_updates", False),
    )
    return ctx.balance


def _ensure_ml(cfg: dict) -> None:
    """Attempt to load the mean_bot ML model or disable ML."""
    if not cfg.get("ml_enabled", True):
        return
    try:  # pragma: no cover - best effort
        from coinTrader_Trainer.ml_trainer import load_model

        load_model("mean_bot")
    except Exception as exc:  # pragma: no cover - missing trainer or model
        cfg["ml_enabled"] = False
        logger.warning("Machine learning unavailable, disabling ml_enabled: %s", exc)


def _emit_timing(
    symbol_t: float,
    ohlcv_t: float,
    analyze_t: float,
    total_t: float,
    metrics_path: Path | None = None,
    ohlcv_fetch_latency: float = 0.0,
    execution_latency: float = 0.0,
) -> None:
    """Log timing information and optionally append to metrics CSV."""
    logger.info(
        "\u23f1\ufe0f Cycle timing - Symbols: %.2fs, OHLCV: %.2fs, Analyze: %.2fs, Total: %.2fs",
        symbol_t,
        ohlcv_t,
        analyze_t,
        total_t,
    )
    if metrics_path:
        log_cycle_metrics(
            symbol_t,
            ohlcv_t,
            analyze_t,
            total_t,
            ohlcv_fetch_latency,
            execution_latency,
            metrics_path,
        )


def load_config() -> dict:
    """Load YAML configuration for the bot."""
    with open(CONFIG_PATH) as f:
        logger.info("Loading config from %s", CONFIG_PATH)
        data = yaml.safe_load(f) or {}

    _ensure_ml(data)

    strat_dir = CONFIG_PATH.parent.parent / "config" / "strategies"
    trend_file = strat_dir / "trend_bot.yaml"
    if trend_file.exists():
        with open(trend_file) as sf:
            overrides = yaml.safe_load(sf) or {}
        trend_cfg = data.get("trend", {})
        if isinstance(trend_cfg, dict):
            trend_cfg.update(overrides)
        else:
            trend_cfg = overrides
        data["trend"] = trend_cfg

    if "symbol" in data:
        data["symbol"] = fix_symbol(data["symbol"])
    if "symbols" in data:
        data["symbols"] = [fix_symbol(s) for s in data.get("symbols") or []]

    onchain_syms = None
    if "onchain_symbols" in data:
        onchain_syms = data.get("onchain_symbols")
    elif "solana_symbols" in data:
        onchain_syms = data.get("solana_symbols")

    if onchain_syms is not None:
        data["onchain_symbols"] = [fix_symbol(s) for s in onchain_syms or []]
    try:
        if hasattr(ScannerConfig, "model_validate"):
            ScannerConfig.model_validate(data)
        else:  # pragma: no cover - for Pydantic < 2
            ScannerConfig.parse_obj(data)
    except ValidationError as exc:
        print("Invalid configuration:\n", exc)
        raise SystemExit(1)

    try:
        raw_scanner = data.get("solana_scanner", {}) or {}
        if hasattr(SolanaScannerConfig, "model_validate"):
            scanner = SolanaScannerConfig.model_validate(raw_scanner)
        else:  # pragma: no cover - for Pydantic < 2
            scanner = SolanaScannerConfig.parse_obj(raw_scanner)
        data["solana_scanner"] = scanner.dict()
    except ValidationError as exc:
        print("Invalid configuration (solana_scanner):\n", exc)
        raise SystemExit(1)

    try:
        raw_pyth = data.get("pyth", {}) or {}
        if hasattr(PythConfig, "model_validate"):
            pyth_cfg = PythConfig.model_validate(raw_pyth)
        else:  # pragma: no cover - for Pydantic < 2
            pyth_cfg = PythConfig.parse_obj(raw_pyth)
        data["pyth"] = pyth_cfg.dict()
    except ValidationError as exc:
        print("Invalid configuration (pyth):\n", exc)
        raise SystemExit(1)

    return data


def maybe_reload_config(state: dict, config: dict) -> None:
    """Reload configuration when ``state['reload']`` is set."""
    if state.get("reload"):
        new_cfg = load_config()
        config.clear()
        config.update(new_cfg)
        state.pop("reload", None)


def _flatten_config(data: dict, parent: str = "") -> dict:
    """Flatten nested config keys to ENV_STYLE names."""
    flat: dict[str, str] = {}
    for key, value in data.items():
        new_key = f"{parent}_{key}" if parent else key
        if isinstance(value, dict):
            flat.update(_flatten_config(value, new_key))
        else:
            flat[new_key.upper()] = value
    return flat


def reload_config(
    config: dict,
    ctx: BotContext,
    risk_manager: RiskManager,
    rotator: PortfolioRotator,
    position_guard: OpenPositionGuard,
    *,
    force: bool = False,
) -> None:
    """Reload the YAML config and update dependent objects."""
    global _LAST_CONFIG_MTIME

    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        mtime = _LAST_CONFIG_MTIME

    if not force and mtime == _LAST_CONFIG_MTIME:
        return

    new_config = load_config()
    _LAST_CONFIG_MTIME = mtime

    config.clear()
    config.update(new_config)
    ctx.config = config

    # Reset cached symbols when configuration changes to ensure
    # symbol selections reflect the latest settings
    symbol_utils._cached_symbols = None
    symbol_utils._last_refresh = 0.0

    rotator.config = config.get("portfolio_rotation", rotator.config)
    position_guard.max_open_trades = config.get(
        "max_open_trades", position_guard.max_open_trades
    )
    logger.info(
        "OpenPositionGuard limit set to %d trades",
        position_guard.max_open_trades,
    )

    cooldown_configure(config.get("min_cooldown", 0))
    market_loader_configure(
        config.get("ohlcv_timeout", 120),
        config.get("max_ohlcv_failures", 3),
        config.get("max_ws_limit", 50),
        config.get("telegram", {}).get("status_updates", True),
        max_concurrent=config.get("max_concurrent_ohlcv"),
        gecko_limit=config.get("gecko_limit"),
    )

    volume_ratio = 0.01 if config.get("testing_mode") else 1.0
    risk_params = {**config.get("risk", {})}
    risk_params.update(config.get("sentiment_filter", {}))
    risk_params.update(config.get("volatility_filter", {}))
    risk_params["symbol"] = config.get("symbol", "")
    risk_params["trade_size_pct"] = config.get("trade_size_pct", 0.1)
    risk_params["strategy_allocation"] = config.get("strategy_allocation", {})
    risk_params["volume_threshold_ratio"] = config.get("risk", {}).get(
        "volume_threshold_ratio", 0.05
    )
    risk_params["atr_period"] = config.get("risk", {}).get("atr_period", 14)
    risk_params["stop_loss_atr_mult"] = config.get("risk", {}).get(
        "stop_loss_atr_mult", 2.0
    )
    risk_params["take_profit_atr_mult"] = config.get("risk", {}).get(
        "take_profit_atr_mult", 4.0
    )
    risk_params["volume_ratio"] = volume_ratio
    risk_manager.config = RiskConfig(**risk_params)


async def _ws_ping_loop(exchange: object, interval: float) -> None:
    """Periodically send WebSocket ping messages."""
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                ping = getattr(exchange, "ping", None)
                if ping is None:
                    continue
                is_coro = asyncio.iscoroutinefunction(ping)
                clients = getattr(exchange, "clients", None)
                if isinstance(clients, dict):
                    if clients:
                        for client in list(clients.values()):
                            if is_coro:
                                await ping(client)
                            else:
                                await asyncio.to_thread(ping, client)
                            logger.debug("Sent WebSocket ping to Kraken")
                    else:
                        continue
                else:
                    if is_coro:
                        await ping()
                    else:
                        await asyncio.to_thread(ping)
                    logger.debug("Sent WebSocket ping to Kraken")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - ping failures
                logger.error("WebSocket ping failed: %s", exc, exc_info=True)
    except asyncio.CancelledError:
        pass


async def registry_update_loop(interval_minutes: float) -> None:
    """Periodically refresh the Solana token registry."""
    from crypto_bot.utils import token_registry as registry

    while True:
        try:
            mapping = await registry.load_token_mints(force_refresh=True)
            if mapping:
                registry.set_token_mints({**registry.TOKEN_MINTS, **mapping})
        except asyncio.CancelledError:
            break
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Token registry update error: %s", exc)
        await asyncio.sleep(interval_minutes * 60)


async def initial_scan(
    exchange: object,
    config: dict,
    state: SessionState,
    notifier: TelegramNotifier | None = None,
) -> None:
    """Populate OHLCV and regime caches before trading begins."""

    ranked, onchain_symbols = await get_filtered_symbols(exchange, config)
    symbols = [s for s, _ in ranked]
    top_n = int(config.get("scan_deep_top", 50))
    symbols = symbols[:top_n]
    for sym in onchain_symbols:
        if sym not in symbols:
            symbols.append(sym)
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return

    batch_size = int(config.get("symbol_batch_size", 10))
    total = len(symbols)
    processed = 0

    sf = config.get("symbol_filter", {})
    ohlcv_batch_size = config.get("ohlcv_batch_size")
    if ohlcv_batch_size is None:
        ohlcv_batch_size = sf.get("ohlcv_batch_size")
    scan_limit = int(
        sf.get("initial_history_candles", config.get("scan_lookback_limit", 50))
    )

    tfs = sf.get("initial_timeframes", config.get("timeframes", ["1h"]))
    tf_sec = timeframe_seconds(None, min(tfs, key=lambda t: timeframe_seconds(None, t)))
    lookback_since = int(time.time() * 1000 - scan_limit * tf_sec * 1000)

    history_since = int((time.time() - 365 * 86400) * 1000)
    deep_limit = int(
        sf.get(
            "initial_history_candles",
            config.get(
                "scan_deep_limit",
                config.get("scan_lookback_limit", 50) * 10,
            ),
        )
    )
    logger.info(
        "Loading deep OHLCV history starting %s",
        datetime.utcfromtimestamp(history_since / 1000).isoformat(),
    )

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]

        async with OHLCV_LOCK:
            state.df_cache = await update_multi_tf_ohlcv_cache(
                exchange,
                state.df_cache,
                batch,
                {**config, "timeframes": tfs},
                limit=deep_limit,
                start_since=history_since,
                use_websocket=False,
                force_websocket_history=config.get("force_websocket_history", False),
                max_concurrent=config.get("max_concurrent_ohlcv"),
                notifier=notifier,
                priority_queue=symbol_priority_queue,
                batch_size=ohlcv_batch_size,
            )

            state.regime_cache = await update_regime_tf_cache(
                exchange,
                state.regime_cache,
                batch,
                {**config, "timeframes": tfs},
                limit=scan_limit,
                start_since=lookback_since,
                use_websocket=False,
                force_websocket_history=config.get("force_websocket_history", False),
                max_concurrent=config.get("max_concurrent_ohlcv"),
                notifier=notifier,
                df_map=state.df_cache,
                batch_size=ohlcv_batch_size,
            )
        logger.info("Deep historical OHLCV loaded for %d symbols", len(batch))

        processed += len(batch)
        pct = processed / total * 100
        logger.info("Initial scan %.1f%% complete", pct)
        if notifier and config.get("telegram", {}).get("status_updates", True):
            notifier.notify(f"Initial scan {pct:.1f}% complete")

    return


async def fetch_candidates(ctx: BotContext) -> None:
    """Gather symbols for this cycle and build the evaluation batch."""
    t0 = time.perf_counter()

    global symbol_priority_queue

    sf = ctx.config.setdefault("symbol_filter", {})

    if (
        not ctx.config.get("symbols")
        and not ctx.config.get("onchain_symbols")
        and not ctx.config.get("symbol")
    ):
        ctx.config["symbol"] = "BTC/USDT"
    orig_min_volume = sf.get("min_volume_usd")
    orig_volume_pct = sf.get("volume_percentile")

    pump = is_market_pumping(
        (ctx.config.get("symbols") or [ctx.config.get("symbol")])
        + ctx.config.get("onchain_symbols", []),
        ctx.df_cache,
        ctx.config.get("timeframe", "1h"),
    )
    if pump:
        sf["min_volume_usd"] = 500
        sf["volume_percentile"] = 5
        if ctx.risk_manager:
            weights = ctx.config.get("strategy_allocation", {}).copy()
            weights["sniper_solana"] = 0.6
            ctx.config["strategy_allocation"] = weights
            ctx.risk_manager.update_allocation(weights)
        ctx.config["mode"] = "auto"

    try:
        symbols, onchain_syms = await get_filtered_symbols(ctx.exchange, ctx.config)
    finally:
        if pump:
            sf["min_volume_usd"] = orig_min_volume
            sf["volume_percentile"] = orig_volume_pct

    # Always include major benchmark pairs
    symbols.extend([("BTC/USDT", 10.0), ("SOL/USDC", 10.0)])

    ctx.timing["symbol_time"] = time.perf_counter() - t0

    solana_tokens: list[str] = list(onchain_syms)
    sol_cfg = ctx.config.get("solana_scanner", {})

    regime = "unknown"
    try:
        regime = await get_market_regime(ctx)
    except Exception:  # pragma: no cover - safety
        pass

    if regime == "trending" and ctx.config.get("arbitrage_enabled", True):
        try:
            if ctx.secondary_exchange:
                arb_syms = await scan_cex_arbitrage(
                    ctx.exchange, ctx.secondary_exchange, ctx.config
                )
            else:
                arb_syms = await scan_arbitrage(ctx.exchange, ctx.config)
            symbols.extend((s, 2.0) for s in arb_syms)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Arbitrage scan error: %s", exc)

    if regime == "volatile":
        symbols.extend((s, 0.0) for s in onchain_syms)

    if regime == "volatile" and sol_cfg.get("enabled"):
        try:
            new_tokens = await get_solana_new_tokens(sol_cfg)
            solana_tokens.extend(new_tokens)
            symbols.extend((m, 0.0) for m in new_tokens)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Solana scanner failed: %s", exc)

    symbol_names = [s for s, _ in symbols]
    avg_atr = compute_average_atr(
        symbol_names, ctx.df_cache, ctx.config.get("timeframe", "1h")
    )
    adaptive_cfg = ctx.config.get("adaptive_scan", {})
    if adaptive_cfg.get("enabled"):
        baseline = adaptive_cfg.get("atr_baseline", 0.01)
        max_factor = adaptive_cfg.get("max_factor", 2.0)
        volatility_factor = min(max_factor, max(1.0, avg_atr / baseline))
    else:
        volatility_factor = 1.0
    ctx.volatility_factor = volatility_factor

    total_available = len(
        (ctx.config.get("symbols") or [ctx.config.get("symbol")])
        + ctx.config.get("onchain_symbols", [])
    )
    ctx.timing["symbol_filter_ratio"] = (
        len(symbols) / total_available if total_available else 1.0
    )

    base_size = ctx.config.get("symbol_batch_size", 10)
    batch_size = int(base_size * volatility_factor)
    async with QUEUE_LOCK:
        if not symbol_priority_queue:
            all_scores = symbols + [(s, 0.0) for s in onchain_syms]
            symbol_priority_queue = build_priority_queue(all_scores)
        if onchain_syms:
            for sym in reversed(onchain_syms):
                symbol_priority_queue.appendleft(sym)
        if solana_tokens:
            for sym in reversed(solana_tokens):
                symbol_priority_queue.appendleft(sym)
        if len(symbol_priority_queue) < batch_size:
            symbol_priority_queue.extend(
                build_priority_queue(symbols + [(s, 0.0) for s in onchain_syms])
            )
        ctx.current_batch = [
            symbol_priority_queue.popleft()
            for _ in range(min(batch_size, len(symbol_priority_queue)))
        ]


async def scan_arbitrage(exchange: object, config: dict) -> list[str]:
    """Return symbols with profitable Solana arbitrage opportunities."""
    pairs: list[str] = config.get("arbitrage_pairs", [])
    if not pairs:
        return []

    try:
        from crypto_bot.utils.market_loader import fetch_geckoterminal_ohlcv
    except Exception:
        fetch_geckoterminal_ohlcv = None

    gecko_prices: dict[str, float] = {}
    if fetch_geckoterminal_ohlcv:
        for sym in pairs:
            try:
                res = await fetch_geckoterminal_ohlcv(sym, limit=1, return_price=True)
            except Exception:
                res = None
            if res:
                _data, _vol, price = res
                gecko_prices[sym] = price

    remaining = [s for s in pairs if s not in gecko_prices]
    dex_prices: dict[str, float] = gecko_prices.copy()
    if remaining:
        try:
            from crypto_bot.solana import fetch_solana_prices
        except Exception:
            fetch_solana_prices = None
        if fetch_solana_prices:
            dex_prices.update(await fetch_solana_prices(remaining))
    results: list[str] = []
    threshold = float(
        config.get(
            "arbitrage_threshold",
            config.get("grid_bot", {}).get("arbitrage_threshold", 0.0),
        )
    )

    for sym in pairs:
        dex_price = dex_prices.get(sym)
        if not dex_price:
            continue
        try:
            if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ticker", None)):
                ticker = await exchange.fetch_ticker(sym)
            else:
                ticker = await asyncio.to_thread(exchange.fetch_ticker, sym)
        except Exception:
            continue
        cex_price = ticker.get("last") or ticker.get("close")
        if cex_price is None:
            continue
        try:
            cex_val = float(cex_price)
        except Exception:
            continue
        if cex_val <= 0:
            continue
        diff = abs(dex_price - cex_val) / cex_val
        if diff >= threshold:
            results.append(sym)
    return results


async def scan_cex_arbitrage(
    primary: object, secondary: object, config: dict
) -> list[str]:
    """Return symbols with profitable arbitrage between two centralized exchanges."""
    pairs: list[str] = config.get("arbitrage_pairs", [])
    if not pairs:
        return []

    results: list[str] = []
    threshold = float(config.get("arbitrage_threshold", 0.0))
    for sym in pairs:
        try:
            if asyncio.iscoroutinefunction(getattr(primary, "fetch_ticker", None)):
                t1 = await primary.fetch_ticker(sym)
            else:
                t1 = await asyncio.to_thread(primary.fetch_ticker, sym)
            if asyncio.iscoroutinefunction(getattr(secondary, "fetch_ticker", None)):
                t2 = await secondary.fetch_ticker(sym)
            else:
                t2 = await asyncio.to_thread(secondary.fetch_ticker, sym)
        except Exception:
            continue

        p1 = t1.get("last") or t1.get("close")
        p2 = t2.get("last") or t2.get("close")
        if p1 is None or p2 is None:
            continue
        try:
            f1 = float(p1)
            f2 = float(p2)
        except Exception:
            continue
        if f1 <= 0:
            continue
        diff = abs(f1 - f2) / f1
        if diff >= threshold:
            results.append(sym)
    return results


async def update_caches(ctx: BotContext) -> None:
    """Update OHLCV and regime caches for the current symbol batch."""
    batch = ctx.current_batch
    if not batch:
        return

    start = time.perf_counter()
    timeframe = ctx.config.get("timeframe", "1h")
    tf_minutes = int(pd.Timedelta(timeframe).total_seconds() // 60)
    limit = max(150, tf_minutes * 2)
    limit = int(ctx.config.get("cycle_lookback_limit") or limit)
    start_since = int(time.time() * 1000 - limit * tf_minutes * 60 * 1000)

    ohlcv_batch_size = ctx.config.get("ohlcv_batch_size")
    if ohlcv_batch_size is None:
        ohlcv_batch_size = ctx.config.get("symbol_filter", {}).get("ohlcv_batch_size")

    max_concurrent = ctx.config.get("max_concurrent_ohlcv")
    if isinstance(max_concurrent, (int, float)):
        max_concurrent = int(max_concurrent)
        if ctx.volatility_factor > 5:
            max_concurrent = max(1, max_concurrent // 2)

    async with OHLCV_LOCK:
        try:
            ctx.df_cache = await update_multi_tf_ohlcv_cache(
                ctx.exchange,
                ctx.df_cache,
                batch,
                ctx.config,
                limit=limit,
                use_websocket=ctx.config.get("use_websocket", False),
                force_websocket_history=ctx.config.get(
                    "force_websocket_history", False
                ),
                max_concurrent=max_concurrent,
                notifier=(
                    ctx.notifier
                    if ctx.config.get("telegram", {}).get("status_updates", True)
                    else None
                ),
                priority_queue=symbol_priority_queue,
                batch_size=ohlcv_batch_size,
            )
        except Exception as exc:
            logger.warning("WS OHLCV failed: %s - falling back to REST", exc)
            ctx.df_cache = await update_multi_tf_ohlcv_cache(
                ctx.exchange,
                ctx.df_cache,
                batch,
                ctx.config,
                limit=limit,
                start_since=start_since,
                use_websocket=False,
                force_websocket_history=ctx.config.get(
                    "force_websocket_history", False
                ),
                max_concurrent=max_concurrent,
                notifier=(
                    ctx.notifier
                    if ctx.config.get("telegram", {}).get("status_updates", True)
                    else None
                ),
                priority_queue=symbol_priority_queue,
                batch_size=ohlcv_batch_size,
            )
        ctx.regime_cache = await update_regime_tf_cache(
            ctx.exchange,
            ctx.regime_cache,
            batch,
            ctx.config,
            limit=limit,
            use_websocket=ctx.config.get("use_websocket", False),
            force_websocket_history=ctx.config.get("force_websocket_history", False),
            max_concurrent=max_concurrent,
            notifier=(
                ctx.notifier
                if ctx.config.get("telegram", {}).get("status_updates", True)
                else None
            ),
            df_map=ctx.df_cache,
            batch_size=ohlcv_batch_size,
        )

    filtered_batch: list[str] = []
    for sym in batch:
        df = ctx.df_cache.get(timeframe, {}).get(sym)
        count = len(df) if isinstance(df, pd.DataFrame) else 0
        logger.info("%s OHLCV: %d candles", sym, count)
        if count == 0:
            logger.warning("No OHLCV data for %s; skipping analysis", sym)
            continue
        filtered_batch.append(sym)

    ctx.current_batch = filtered_batch

    vol_thresh = ctx.config.get("bounce_scalper", {}).get("vol_zscore_threshold")
    if vol_thresh is not None:
        status_updates = ctx.config.get("telegram", {}).get("status_updates", True)
        for sym in batch:
            df = ctx.df_cache.get(timeframe, {}).get(sym)
            if df is None or df.empty or "volume" not in df:
                continue
            vols = df["volume"].to_numpy(dtype=float)
            mean = float(np.mean(vols)) if len(vols) else 0.0
            std = float(np.std(vols))
            if std <= 0:
                continue
            z_scores = (vols - mean) / std
            z_max = float(np.max(z_scores))
            if z_max > vol_thresh:
                async with QUEUE_LOCK:
                    symbol_priority_queue.appendleft(sym)
                msg = f"Volume spike priority for {sym}: z={z_max:.2f}"
                logger.info(msg)
                if status_updates and ctx.notifier:
                    ctx.notifier.notify(msg)

    if ctx.config.get("use_websocket", True):
        timeframe = ctx.config.get("timeframe", "1h")
        try:
            # Subscribe to WS for live candles
            await ctx.exchange.watch_ohlcv(batch, timeframe)
        except Exception as exc:  # pragma: no cover - network
            logger.warning("WS subscribe failed: %s", exc)

    ctx.timing["ohlcv_fetch_latency"] = time.perf_counter() - start


async def enrich_with_pyth(ctx: BotContext) -> None:
    """Update cached OHLCV using the latest Pyth prices."""
    batch = ctx.current_batch
    if not batch:
        return

    async with aiohttp.ClientSession() as session:
        for sym in batch:
            quote = sym.split("/")[-1]
            allowed = ctx.config.get("pyth_quotes", ["USDC"])
            if quote not in allowed:
                continue
            base = sym.split("/")[0]
            try:
                url = f"https://hermes.pyth.network/v2/price_feeds?query={base}"
                async with session.get(url, timeout=10) as resp:
                    feeds = await resp.json()
            except Exception:
                continue

            feed_id = None
            for item in feeds:
                attrs = item.get("attributes", {})
                if attrs.get("base") == base and attrs.get("quote_currency") == "USD":
                    feed_id = item.get("id")
                    break
            if not feed_id:
                continue

            try:
                url = (
                    "https://hermes.pyth.network/api/latest_price_feeds?ids[]="
                    f"{feed_id}"
                )
                async with session.get(url, timeout=10) as resp:
                    data = await resp.json()
            except Exception:
                continue

            if not data:
                continue

            price_info = data[0].get("price")
            if not price_info:
                continue

            price = float(price_info.get("price", 0)) * (
                10 ** price_info.get("expo", 0)
            )

            async with OHLCV_LOCK:
                for cache in ctx.df_cache.values():
                    df = cache.get(sym)
                    if df is not None and not df.empty:
                        df.loc[df.index[-1], "close"] = price


async def analyse_batch(ctx: BotContext) -> None:
    """Run signal analysis on the current batch."""
    batch = ctx.current_batch
    if not batch:
        logger.info("analyse_batch called with empty batch")
        ctx.analysis_results = []
        return

    logger.info(
        "analyse_batch starting with %d symbols: %s",
        len(batch),
        batch,
    )

    base_tf = ctx.config.get("timeframe", "1h")

    tasks = []
    mode = ctx.config.get("mode", "cex")
    for sym in batch:
        df_map = {tf: c.get(sym) for tf, c in ctx.df_cache.items()}
        for tf, cache in ctx.regime_cache.items():
            df_map[tf] = cache.get(sym)
        df = df_map.get(base_tf)
        logger.info(
            "DF len for %s: %d",
            sym,
            len(df) if isinstance(df, pd.DataFrame) else 0,
        )
        tasks.append(
            analyze_symbol(
                sym,
                df_map,
                mode,
                ctx.config,
                ctx.notifier,
                mempool_monitor=ctx.mempool_monitor,
                mempool_cfg=ctx.mempool_cfg,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    ctx.analysis_results = []
    for sym, res in zip(batch, results):
        if isinstance(res, Exception):
            logger.error("Analysis failed for %s: %s", sym, res)
        else:
            ctx.analysis_results.append(res)

    global UNKNOWN_COUNT, TOTAL_ANALYSES
    for res in ctx.analysis_results:
        if res.get("skip"):
            continue
        logger.info(
            "Analysis for %s: regime=%s, score=%s",
            res["symbol"],
            res["regime"],
            res["score"],
        )
        TOTAL_ANALYSES += 1
        if res.get("regime") == "unknown":
            UNKNOWN_COUNT += 1

    logger.info(
        "analyse_batch produced %d results (%d skipped)",
        len(ctx.analysis_results),
        sum(1 for r in ctx.analysis_results if r.get("skip")),
    )


async def execute_signals(ctx: BotContext) -> None:
    """Open trades for qualified analysis results."""
    results = getattr(ctx, "analysis_results", [])
    if not results:
        logger.info("No analysis results to act on")
        return

    # Filter and prioritize by score
    orig_results = results
    results = []
    skipped_syms: list[str] = []
    no_dir_syms: list[str] = []
    low_score: list[str] = []
    for r in orig_results:
        if r.get("skip"):
            skipped_syms.append(r.get("symbol"))
            continue
        if r.get("direction") == "none":
            no_dir_syms.append(r.get("symbol"))
            continue
        min_req = r.get("min_confidence", ctx.config.get("min_confidence_score", 0.0))
        score = r.get("score", 0.0)
        if score < min_req:
            low_score.append(f"{r.get('symbol')}({score:.2f}<{min_req:.2f})")
            continue
        results.append(r)

    logger.debug(
        "Candidate scoring: %d/%d met the minimum score; low_score=%s skip=%s no_direction=%s",
        len(results),
        len(orig_results),
        low_score,
        skipped_syms,
        no_dir_syms,
    )

    if not results:
        logger.info("All signals filtered out - nothing actionable")
        if ctx.notifier and ctx.config.get("telegram", {}).get("trade_updates", True):
            ctx.notifier.notify("No symbols qualified for trading")
        return

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_n = ctx.config.get("top_n_symbols", 10)
    executed = 0

    for candidate in results[:top_n]:
        logger.info("Analysis result: %s", candidate)
        max_trades = ctx.position_guard.max_open_trades if ctx.position_guard else 0
        logger.debug("Open trades: %d / %d", len(ctx.positions), max_trades)
        if not ctx.position_guard or not ctx.position_guard.can_open(ctx.positions):
            logger.info(
                "Max open trades reached (%d/%d); skipping remaining signals",
                len(ctx.positions),
                max_trades,
            )
            break
        sym = candidate["symbol"]
        logger.info("[EVAL] evaluating %s", sym)
        outcome_reason = ""
        if ctx.balance <= 0:
            old_balance = ctx.balance
            latest = await refresh_balance(ctx)
            logger.info(
                "Balance was %.2f; refreshed to %.2f for %s",
                old_balance,
                latest,
                sym,
            )
        if sym in ctx.positions:
            outcome_reason = "existing position"
            logger.info("[EVAL] %s -> %s", sym, outcome_reason)
            continue

        df = candidate["df"]
        price = df["close"].iloc[-1]
        score = candidate.get("score", 0.0)
        strategy = candidate.get("name", "")
        allowed, reason = ctx.risk_manager.allow_trade(df, strategy, sym)
        if not allowed:
            outcome_reason = f"blocked: {reason}"
            logger.info("[EVAL] %s -> %s", sym, outcome_reason)
            continue

        probs = candidate.get("probabilities", {})
        reg_prob = float(probs.get(candidate.get("regime"), 0.0))
        size = ctx.risk_manager.position_size(
            reg_prob,
            ctx.balance,
            df,
            atr=candidate.get("atr"),
            price=price,
            name=strategy,
        )
        if size <= 0:
            await refresh_balance(ctx)
            size = ctx.risk_manager.position_size(
                reg_prob,
                ctx.balance,
                df,
                atr=candidate.get("atr"),
                price=price,
                name=strategy,
            )
            if size <= 0:
                outcome_reason = f"size {size:.4f}"
                logger.info("[EVAL] %s -> %s", sym, outcome_reason)
                continue

        if not ctx.risk_manager.can_allocate(strategy, size, ctx.balance):
            logger.info(
                "Insufficient capital to allocate %.4f for %s via %s",
                size,
                sym,
                strategy,
            )
            outcome_reason = "insufficient capital"
            logger.info("[EVAL] %s -> %s", sym, outcome_reason)
            continue

        amount = size / price if price > 0 else 0.0
        side = direction_to_side(candidate["direction"])
        if side == "sell" and not ctx.config.get("allow_short", False):
            outcome_reason = "short selling disabled"
            logger.info("[EVAL] %s -> %s", sym, outcome_reason)
            continue
        start_exec = time.perf_counter()
        executed_via_sniper = False
        if sym.endswith("/USDC"):
            from crypto_bot.solana import sniper_solana

            sol_score, _ = sniper_solana.generate_signal(df)
            if sol_score > 0.7:
                from crypto_bot.solana_trading import sniper_trade

                base, quote = sym.split("/")
                task = asyncio.create_task(
                    sniper_trade(
                        ctx.config.get("wallet_address", ""),
                        quote,
                        base,
                        size,
                        dry_run=ctx.config.get("execution_mode") == "dry_run",
                        slippage_bps=ctx.config.get("solana_slippage_bps", 50),
                        notifier=ctx.notifier,
                    )
                )
                SNIPER_TASKS.add(task)
                task.add_done_callback(SNIPER_TASKS.discard)
                executed_via_sniper = True

        if not executed_via_sniper:
            order = await cex_trade_async(
                ctx.exchange,
                ctx.ws_client,
                sym,
                side,
                amount,
                ctx.notifier,
                dry_run=ctx.config.get("execution_mode") == "dry_run",
                use_websocket=ctx.config.get("use_websocket", False),
                config=ctx.config,
            )
            if order:
                executed += 1
                logger.info("[EVAL] %s -> executed %s %.4f", sym, side, amount)
                take_profit = None
                if strategy == "bounce_scalper":
                    depth = int(ctx.config.get("liquidity_depth", 10))
                    book = await fetch_order_book_async(ctx.exchange, sym, depth)
                    dist = _closest_wall_distance(book, price, side)
                    if dist is not None:
                        take_profit = dist * 0.8
                ctx.risk_manager.register_stop_order(
                    order,
                    strategy=strategy,
                    symbol=sym,
                    entry_price=price,
                    confidence=score,
                    direction=side,
                    take_profit=take_profit,
                    regime=candidate.get("regime"),
                )
        else:
            executed += 1
            logger.info("[EVAL] %s -> executed %s %.4f", sym, side, amount)
        ctx.timing["execution_latency"] = max(
            ctx.timing.get("execution_latency", 0.0),
            time.perf_counter() - start_exec,
        )

        if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
            ctx.paper_wallet.open(sym, side, amount, price)
            ctx.balance = ctx.paper_wallet.balance
        ctx.risk_manager.allocate_capital(strategy, size)
        ctx.positions[sym] = {
            "side": side,
            "entry_price": price,
            "entry_time": datetime.utcnow().isoformat(),
            "regime": candidate.get("regime"),
            "strategy": strategy,
            "confidence": score,
            "pnl": 0.0,
            "size": amount,
            "trailing_stop": 0.0,
            "highest_price": price,
            "dca_count": 0,
        }
        try:
            log_position(
                sym,
                side,
                amount,
                price,
                price,
                ctx.balance,
            )
        except Exception:
            pass

        await refresh_balance(ctx)

        if strategy == "micro_scalp":
            asyncio.create_task(_monitor_micro_scalp_exit(ctx, sym))

    if executed == 0:
        logger.info(
            "No trades executed from %d candidate signals", len(results[:top_n])
        )


async def handle_exits(ctx: BotContext) -> None:
    """Check open positions for exit conditions."""
    tf = ctx.config.get("timeframe", "1h")
    tf_cache = ctx.df_cache.get(tf, {})
    for sym, pos in list(ctx.positions.items()):
        df = tf_cache.get(sym)
        if df is None or df.empty or "close" not in df:
            continue
        current_price = float(df["close"].iloc[-1])
        pnl_pct = ((current_price - pos["entry_price"]) / pos["entry_price"]) * (
            1 if pos["side"] == "buy" else -1
        )
        if pnl_pct >= ctx.config.get("exit_strategy", {}).get("min_gain_to_trail", 0):
            if current_price > pos.get("highest_price", pos["entry_price"]):
                pos["highest_price"] = current_price
            pos["trailing_stop"] = calculate_trailing_stop(
                pd.Series([pos.get("highest_price", current_price)]),
                ctx.config.get("exit_strategy", {}).get("trailing_stop_pct", 0.1),
            )

        # DCA before evaluating exit conditions
        dca_cfg = ctx.config.get("dca", {})
        dca_score, dca_dir = dca_bot.generate_signal(df)
        if (
            dca_score > 0
            and pos.get("dca_count", 0) < dca_cfg.get("max_entries", 0)
            and (
                (pos["side"] == "buy" and dca_dir == "long")
                or (pos["side"] == "sell" and dca_dir == "short")
            )
        ):
            add_amount = pos["size"] * dca_cfg.get("size_multiplier", 1.0)
            add_value = add_amount * current_price
            if ctx.risk_manager.can_allocate(
                pos.get("strategy", ""), add_value, ctx.balance
            ):
                await cex_trade_async(
                    ctx.exchange,
                    ctx.ws_client,
                    sym,
                    pos["side"],
                    add_amount,
                    ctx.notifier,
                    dry_run=ctx.config.get("execution_mode") == "dry_run",
                    use_websocket=ctx.config.get("use_websocket", False),
                    config=ctx.config,
                )
                if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
                    try:
                        ctx.paper_wallet.open(
                            sym, pos["side"], add_amount, current_price
                        )
                        ctx.balance = ctx.paper_wallet.balance
                    except Exception:
                        pass
                await refresh_balance(ctx)
                ctx.risk_manager.allocate_capital(pos.get("strategy", ""), add_value)
                prev_amount = pos["size"]
                pos["size"] += add_amount
                pos["entry_price"] = (
                    pos["entry_price"] * prev_amount + current_price * add_amount
                ) / pos["size"]
                pos["dca_count"] = pos.get("dca_count", 0) + 1
                ctx.risk_manager.update_stop_order(pos["size"], symbol=sym)
        exit_signal, new_stop = should_exit(
            df,
            current_price,
            pos.get("trailing_stop", 0.0),
            ctx.config,
            ctx.risk_manager,
        )
        pos["trailing_stop"] = new_stop
        if exit_signal:
            await cex_trade_async(
                ctx.exchange,
                ctx.ws_client,
                sym,
                opposite_side(pos["side"]),
                pos["size"],
                ctx.notifier,
                dry_run=ctx.config.get("execution_mode") == "dry_run",
                use_websocket=ctx.config.get("use_websocket", False),
                config=ctx.config,
            )
            if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
                try:
                    ctx.paper_wallet.close(sym, pos["size"], current_price)
                    ctx.balance = ctx.paper_wallet.balance
                except Exception:
                    pass
            await refresh_balance(ctx)
            realized_pnl = (current_price - pos["entry_price"]) * pos["size"]
            if pos["side"] == "sell":
                realized_pnl = -realized_pnl
            pnl_logger.log_pnl(
                pos.get("strategy", ""),
                sym,
                pos["entry_price"],
                current_price,
                realized_pnl,
                pos.get("confidence", 0.0),
                pos["side"],
            )
            regime_pnl_tracker.log_trade(
                pos.get("regime", ""), pos.get("strategy", ""), realized_pnl
            )
            ctx.risk_manager.deallocate_capital(
                pos.get("strategy", ""), pos["size"] * pos["entry_price"]
            )
            ctx.positions.pop(sym, None)
            try:
                log_position(
                    sym,
                    pos["side"],
                    pos["size"],
                    pos["entry_price"],
                    current_price,
                    ctx.balance,
                )
            except Exception:
                pass
            try:
                pnl = (current_price - pos["entry_price"]) / pos["entry_price"]
                if pos["side"] == "sell":
                    pnl = -pnl
                regime_pnl_tracker.log_trade(
                    pos.get("regime", ""), pos.get("strategy", ""), pnl
                )
            except Exception:
                pass
        else:
            score, direction = dca_bot.generate_signal(df)
            dca_cfg = ctx.config.get("dca", {})
            max_entries = dca_cfg.get("max_entries", 0)
            size_pct = dca_cfg.get("size_pct", 1.0)
            if (
                pos.get("dca_count", 0) < max_entries
                and (
                    (pos["side"] == "buy" and direction == "long")
                    or (pos["side"] == "sell" and direction == "short")
                )
                and ctx.risk_manager.capital_tracker.can_allocate(
                    pos.get("strategy", ""),
                    pos["size"] * size_pct * current_price,
                    ctx.balance,
                )
            ):
                new_size = pos["size"] * size_pct
                await cex_trade_async(
                    ctx.exchange,
                    ctx.ws_client,
                    sym,
                    pos["side"],
                    new_size,
                    ctx.notifier,
                    dry_run=ctx.config.get("execution_mode") == "dry_run",
                    use_websocket=ctx.config.get("use_websocket", False),
                    config=ctx.config,
                )
                if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
                    try:
                        ctx.paper_wallet.open(sym, pos["side"], new_size, current_price)
                        ctx.balance = ctx.paper_wallet.balance
                    except Exception:
                        pass
                ctx.risk_manager.allocate_capital(
                    pos.get("strategy", ""), new_size * current_price
                )
                pos["size"] += new_size
                pos["dca_count"] = pos.get("dca_count", 0) + 1
                try:
                    log_position(
                        sym,
                        pos["side"],
                        pos["size"],
                        pos["entry_price"],
                        current_price,
                        ctx.balance,
                    )
                except Exception:
                    pass

            # persist updated fields like 'dca_count'
            ctx.positions[sym] = pos


async def force_exit_all(ctx: BotContext) -> None:
    """Liquidate all open positions immediately."""
    tf = ctx.config.get("timeframe", "1h")
    tf_cache = ctx.df_cache.get(tf, {})
    if not ctx.positions:
        logger.info("No positions to liquidate")
    for sym, pos in list(ctx.positions.items()):
        df = tf_cache.get(sym)
        exit_price = pos["entry_price"]
        if df is not None and not df.empty:
            exit_price = float(df["close"].iloc[-1])

        logger.info("Liquidating %s %.4f @ %.2f", sym, pos["size"], exit_price)

        await cex_trade_async(
            ctx.exchange,
            ctx.ws_client,
            sym,
            opposite_side(pos["side"]),
            pos["size"],
            ctx.notifier,
            dry_run=ctx.config.get("execution_mode") == "dry_run",
            use_websocket=ctx.config.get("use_websocket", False),
            config=ctx.config,
        )

        if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
            try:
                ctx.paper_wallet.close(sym, pos["size"], exit_price)
                ctx.balance = ctx.paper_wallet.balance
            except Exception:
                pass

        await refresh_balance(ctx)
        realized_pnl = (exit_price - pos["entry_price"]) * pos["size"]
        if pos["side"] == "sell":
            realized_pnl = -realized_pnl
        pnl_logger.log_pnl(
            pos.get("strategy", ""),
            sym,
            pos["entry_price"],
            exit_price,
            realized_pnl,
            pos.get("confidence", 0.0),
            pos["side"],
        )
        regime_pnl_tracker.log_trade(
            pos.get("regime", ""), pos.get("strategy", ""), realized_pnl
        )

        ctx.risk_manager.deallocate_capital(
            pos.get("strategy", ""), pos["size"] * pos["entry_price"]
        )
        ctx.positions.pop(sym, None)
        try:
            log_position(
                sym,
                pos["side"],
                pos["size"],
                pos["entry_price"],
                exit_price,
                ctx.balance,
            )
        except Exception:
            pass

    if not ctx.positions and ctx.paper_wallet and ctx.paper_wallet.positions:
        for pid, wpos in list(ctx.paper_wallet.positions.items()):
            sym = wpos.get("symbol") or pid
            df = tf_cache.get(sym)
            exit_price = wpos.get("entry_price", 0.0)
            if df is not None and not df.empty:
                exit_price = float(df["close"].iloc[-1])

            size = wpos.get("size", wpos.get("amount", 0.0))
            try:
                ctx.paper_wallet.close(size, exit_price, pid)
                ctx.balance = ctx.paper_wallet.balance
            except Exception:
                pass

            realized_pnl = (exit_price - wpos.get("entry_price", 0.0)) * size
            if wpos.get("side") == "sell":
                realized_pnl = -realized_pnl
            pnl_logger.log_pnl(
                wpos.get("strategy", ""),
                sym,
                wpos.get("entry_price", 0.0),
                exit_price,
                realized_pnl,
                wpos.get("confidence", 0.0),
                wpos.get("side", ""),
            )
            regime_pnl_tracker.log_trade(
                wpos.get("regime", ""), wpos.get("strategy", ""), realized_pnl
            )

            ctx.risk_manager.deallocate_capital(
                wpos.get("strategy", ""), size * wpos.get("entry_price", 0.0)
            )
            try:
                log_position(
                    sym,
                    wpos.get("side", ""),
                    size,
                    wpos.get("entry_price", 0.0),
                    exit_price,
                    ctx.balance,
                )
            except Exception:
                pass
            logger.info("Liquidated %s %.4f @ %.2f", sym, size, exit_price)

        ctx.paper_wallet.positions.clear()


async def _monitor_micro_scalp_exit(ctx: BotContext, sym: str) -> None:
    """Monitor a micro-scalp trade and exit based on :func:`monitor_price`."""
    pos = ctx.positions.get(sym)
    if not pos:
        return

    tf = ctx.config.get("scalp_timeframe", "1m")

    def feed() -> float:
        df = ctx.df_cache.get(tf, {}).get(sym)
        if df is None or df.empty:
            return pos["entry_price"]
        return float(df["close"].iloc[-1])

    res = await monitor_price(feed, pos["entry_price"], {})
    exit_price = res.get("exit_price", feed())

    await cex_trade_async(
        ctx.exchange,
        ctx.ws_client,
        sym,
        opposite_side(pos["side"]),
        pos["size"],
        ctx.notifier,
        dry_run=ctx.config.get("execution_mode") == "dry_run",
        use_websocket=ctx.config.get("use_websocket", False),
        config=ctx.config,
    )

    if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
        try:
            ctx.paper_wallet.close(sym, pos["size"], exit_price)
            ctx.balance = ctx.paper_wallet.balance
        except Exception:
            pass

    await refresh_balance(ctx)
    realized_pnl = (exit_price - pos["entry_price"]) * pos["size"]
    if pos["side"] == "sell":
        realized_pnl = -realized_pnl
    pnl_logger.log_pnl(
        pos.get("strategy", ""),
        sym,
        pos["entry_price"],
        exit_price,
        realized_pnl,
        pos.get("confidence", 0.0),
        pos["side"],
    )
    regime_pnl_tracker.log_trade(
        pos.get("regime", ""), pos.get("strategy", ""), realized_pnl
    )

    ctx.risk_manager.deallocate_capital(
        pos.get("strategy", ""), pos["size"] * pos["entry_price"]
    )
    ctx.positions.pop(sym, None)
    try:
        log_position(
            sym, pos["side"], pos["size"], pos["entry_price"], exit_price, ctx.balance
        )
    except Exception:
        pass
    try:
        pnl = (exit_price - pos["entry_price"]) / pos["entry_price"]
        if pos["side"] == "sell":
            pnl = -pnl
        regime_pnl_tracker.log_trade(
            pos.get("regime", ""), pos.get("strategy", ""), pnl
        )
    except Exception:
        pass


async def _rotation_loop(
    rotator: PortfolioRotator,
    exchange: object,
    wallet: str,
    state: dict,
    notifier: TelegramNotifier | None,
    check_balance_change: callable,
) -> None:
    """Periodically rotate portfolio holdings."""

    interval = rotator.config.get("interval_days", 7) * 86400
    while True:
        try:
            if state.get("running") and rotator.config.get("enabled"):
                if asyncio.iscoroutinefunction(
                    getattr(exchange, "fetch_balance", None)
                ):
                    bal = await exchange.fetch_balance()
                else:
                    bal = await asyncio.to_thread(exchange.fetch_balance)
                current_balance = (
                    bal.get("USDT", {}).get("free", 0)
                    if isinstance(bal.get("USDT"), dict)
                    else bal.get("USDT", 0)
                )
                check_balance_change(float(current_balance), "external change")
                holdings = {
                    k: (v.get("total") if isinstance(v, dict) else v)
                    for k, v in bal.items()
                }
                await rotator.rotate(exchange, wallet, holdings, notifier)
        except asyncio.CancelledError:
            break
        except Exception as exc:  # pragma: no cover - rotation errors
            logger.error("Rotation loop error: %s", exc, exc_info=True)
        sleep_remaining = interval
        while sleep_remaining > 0:
            sleep_chunk = min(60, sleep_remaining)
            await asyncio.sleep(sleep_chunk)
            sleep_remaining -= sleep_chunk
            if not (rotator.config.get("enabled") and state.get("running")):
                break


async def _main_impl() -> TelegramNotifier:
    """Implementation for running the trading bot."""

    logger.info("Starting bot")
    global UNKNOWN_COUNT, TOTAL_ANALYSES
    config = load_config()
    from crypto_bot.utils.token_registry import (
        TOKEN_MINTS,
        load_token_mints,
        set_token_mints,
    )

    mapping = await load_token_mints()
    if mapping:
        set_token_mints({**TOKEN_MINTS, **mapping})
    onchain_syms = [fix_symbol(s) for s in config.get("onchain_symbols", [])]
    onchain_syms = [f"{s}/USDC" if "/" not in s else s for s in onchain_syms]
    if onchain_syms:
        config["onchain_symbols"] = onchain_syms
    sol_syms = [fix_symbol(s) for s in config.get("solana_symbols", [])]
    sol_syms = [f"{s}/USDC" if "/" not in s else s for s in sol_syms]
    if sol_syms:
        config["onchain_symbols"] = sol_syms
    global _LAST_CONFIG_MTIME
    try:
        _LAST_CONFIG_MTIME = CONFIG_PATH.stat().st_mtime
    except OSError:
        pass
    metrics_path = (
        Path(config.get("metrics_csv")) if config.get("metrics_csv") else None
    )

    async def solana_scan_loop() -> None:
        """Periodically fetch new Solana tokens and queue them."""
        cfg = config.get("solana_scanner", {})
        interval = cfg.get("interval_minutes", 5) * 60
        while True:
            try:
                tokens = await get_solana_new_tokens(cfg)
                if tokens:
                    async with QUEUE_LOCK:
                        for sym in reversed(tokens):
                            symbol_priority_queue.appendleft(sym)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - best effort
                logger.error("Solana scan error: %s", exc)
            await asyncio.sleep(interval)

    volume_ratio = 0.01 if config.get("testing_mode") else 1.0
    cooldown_configure(config.get("min_cooldown", 0))
    status_updates = config.get("telegram", {}).get("status_updates", True)
    market_loader_configure(
        config.get("ohlcv_timeout", 120),
        config.get("max_ohlcv_failures", 3),
        config.get("max_ws_limit", 50),
        status_updates,
        max_concurrent=config.get("max_concurrent_ohlcv"),
        gecko_limit=config.get("gecko_limit"),
    )
    secrets = dotenv_values(ENV_PATH)
    flat_cfg = _flatten_config(config)
    for key, val in secrets.items():
        if key in flat_cfg:
            if flat_cfg[key] != val:
                logger.info(
                    "Overriding %s from .env (config.yaml value: %s)",
                    key,
                    flat_cfg[key],
                )
            else:
                logger.info("Using %s from .env (matches config.yaml)", key)
        else:
            logger.info("Setting %s from .env", key)
    os.environ.update(secrets)

    user = load_or_create()

    status_updates = config.get("telegram", {}).get("status_updates", True)
    balance_updates = config.get("telegram", {}).get("balance_updates", False)

    tg_cfg = {**config.get("telegram", {})}
    if user.get("telegram_token"):
        tg_cfg["token"] = user["telegram_token"]
    if user.get("telegram_chat_id"):
        tg_cfg["chat_id"] = user["telegram_chat_id"]
    if os.getenv("TELE_CHAT_ADMINS"):
        tg_cfg["chat_admins"] = os.getenv("TELE_CHAT_ADMINS")
    status_updates = tg_cfg.get("status_updates", status_updates)
    balance_updates = tg_cfg.get("balance_updates", balance_updates)

    notifier = TelegramNotifier.from_config(tg_cfg)
    if status_updates:
        notifier.notify("🤖 CoinTrader2.0 started")

    mempool_cfg = config.get("mempool_monitor", {})
    mempool_monitor = None
    if mempool_cfg.get("enabled"):
        mempool_monitor = SolanaMempoolMonitor()

    if notifier.token and notifier.chat_id:
        if not send_test_message(notifier.token, notifier.chat_id, "Bot started"):
            logger.warning("Telegram test message failed; check your token and chat ID")

    # allow user-configured exchange to override YAML setting
    if user.get("exchange"):
        config["primary_exchange"] = user["exchange"]

    exchanges = get_exchanges(config)
    primary = (
        config.get("primary_exchange")
        or config.get("exchange")
        or next(iter(exchanges))
    )
    exchange, ws_client = exchanges[primary]
    secondary_exchange = None
    for name, pair in exchanges.items():
        if name != primary:
            secondary_exchange = pair[0]
            break
    if hasattr(exchange, "options"):
        opts = getattr(exchange, "options", {})
        opts["ws"] = {"ping_interval": 10, "ping_timeout": 45}
        exchange.options = opts

    ping_interval = int(config.get("ws_ping_interval", 0) or 0)
    if ping_interval > 0 and hasattr(exchange, "ping"):
        task = asyncio.create_task(_ws_ping_loop(exchange, ping_interval))
        WS_PING_TASKS.add(task)

    if not hasattr(exchange, "load_markets"):
        logger.error("The installed ccxt package is missing or a local stub is in use.")
        if status_updates:
            notifier.notify(
                "❌ ccxt library not found or stubbed; check your installation"
            )
        # Continue startup even if ccxt is missing for testing environments

    if config.get("scan_markets", True) and not config.get("symbols"):
        attempt = 0
        delay = SYMBOL_SCAN_RETRY_DELAY
        discovered: list[str] | None = None
        while attempt < MAX_SYMBOL_SCAN_ATTEMPTS:
            start_scan = time.perf_counter()
            discovered = await load_kraken_symbols(
                exchange,
                config.get("excluded_symbols", []),
                config,
            )
            latency = time.perf_counter() - start_scan
            record_sol_scanner_metrics(len(discovered or []), latency, config)
            if discovered:
                break
            attempt += 1
            if attempt >= MAX_SYMBOL_SCAN_ATTEMPTS:
                break
            logger.warning(
                "Symbol scan empty; retrying in %d seconds (attempt %d/%d)",
                delay,
                attempt + 1,
                MAX_SYMBOL_SCAN_ATTEMPTS,
            )
            if status_updates:
                notifier.notify(
                    f"Symbol scan failed; retrying in {delay}s (attempt {attempt + 1}/{MAX_SYMBOL_SCAN_ATTEMPTS})"
                )
            if inspect.iscoroutinefunction(asyncio.sleep):
                await asyncio.sleep(delay)
            else:  # pragma: no cover - compatibility with patched sleep
                asyncio.sleep(delay)
            delay = min(delay * 2, MAX_SYMBOL_SCAN_DELAY)

        if discovered:
            config["symbols"] = discovered + config.get("onchain_symbols", [])
            onchain_syms = config.get("onchain_symbols", [])
            cex_count = len([s for s in config["symbols"] if s not in onchain_syms])
            logger.info(
                "Loaded %d CEX symbols and %d onchain symbols",
                cex_count,
                len(onchain_syms),
            )
        elif discovered is None:
            cached = load_liquid_pairs()
            if isinstance(cached, list):
                config["symbols"] = cached + config.get("onchain_symbols", [])
                logger.warning("Using cached pairs due to symbol scan failure")
                onchain_syms = config.get("onchain_symbols", [])
                cex_count = len([s for s in config["symbols"] if s not in onchain_syms])
                logger.info(
                    "Loaded %d CEX symbols and %d onchain symbols",
                    cex_count,
                    len(onchain_syms),
                )
            else:
                fallback: list[str] | None = None
                if not PAIR_FILE.exists():
                    rp_cfg = config.get("refresh_pairs", {})
                    min_vol = float(
                        rp_cfg.get("min_quote_volume_usd", DEFAULT_MIN_VOLUME_USD)
                    )
                    top_k = int(rp_cfg.get("top_k", DEFAULT_TOP_K))
                    try:
                        fallback = await refresh_pairs_async(
                            min_vol, top_k, config, force_refresh=True
                        )
                    except Exception as exc:  # pragma: no cover - network errors
                        logger.error("refresh_pairs_async failed: %s", exc)
                if fallback:
                    config["symbols"] = fallback + config.get("onchain_symbols", [])
                    logger.warning("Loaded fresh pairs after scan failure")
                else:
                    logger.error(
                        "No symbols discovered after %d attempts; aborting startup",
                        MAX_SYMBOL_SCAN_ATTEMPTS,
                    )
                    if status_updates:
                        notifier.notify(
                            f"❌ Startup aborted after {MAX_SYMBOL_SCAN_ATTEMPTS} symbol scan attempts"
                        )
                    return notifier
        else:
            logger.error(
                "No symbols discovered after %d attempts; aborting startup",
                MAX_SYMBOL_SCAN_ATTEMPTS,
            )
            if status_updates:
                notifier.notify(
                    f"❌ Startup aborted after {MAX_SYMBOL_SCAN_ATTEMPTS} symbol scan attempts"
                )
            return notifier

    balance_threshold = config.get("balance_change_threshold", 0.01)
    previous_balance = 0.0

    def check_balance_change(new_balance: float, reason: str) -> None:
        nonlocal previous_balance
        delta = new_balance - previous_balance
        if abs(delta) > balance_threshold and notifier:
            notifier.notify(f"Balance changed by {delta:.4f} USDT due to {reason}")
        previous_balance = new_balance

    try:
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
            bal = await exchange.fetch_balance()
        else:
            bal = await asyncio.to_thread(exchange.fetch_balance)
        init_bal = (
            bal.get("USDT", {}).get("free", 0)
            if isinstance(bal.get("USDT"), dict)
            else bal.get("USDT", 0)
        )
        log_balance(float(init_bal))
        last_balance = float(init_bal)
        previous_balance = float(init_bal)
    except Exception as exc:  # pragma: no cover - network
        logger.error("Exchange API setup failed: %s", exc)
        if status_updates:
            err = notifier.notify(f"API error: {exc}")
            if err:
                logger.error("Failed to notify user: %s", err)
        return notifier
    risk_params = {**config.get("risk", {})}
    risk_params.update(config.get("sentiment_filter", {}))
    risk_params.update(config.get("volatility_filter", {}))
    risk_params["symbol"] = config.get("symbol", "")
    risk_params["trade_size_pct"] = config.get("trade_size_pct", 0.1)
    risk_params["strategy_allocation"] = config.get("strategy_allocation", {})
    risk_params["volume_threshold_ratio"] = config.get("risk", {}).get(
        "volume_threshold_ratio", 0.05
    )
    risk_params["atr_period"] = config.get("risk", {}).get("atr_period", 14)
    risk_params["stop_loss_atr_mult"] = config.get("risk", {}).get(
        "stop_loss_atr_mult", 2.0
    )
    risk_params["take_profit_atr_mult"] = config.get("risk", {}).get(
        "take_profit_atr_mult", 4.0
    )
    risk_params["volume_ratio"] = volume_ratio
    risk_config = RiskConfig(**risk_params)
    risk_manager = RiskManager(risk_config)

    paper_wallet = None
    if config.get("execution_mode") == "dry_run":
        try:
            start_bal = float(input("Enter paper trading balance in USDT: "))
        except Exception:
            start_bal = 1000.0
        paper_wallet = PaperWallet(
            start_bal,
            config.get("max_open_trades", 1),
            config.get("allow_short", False),
        )
        log_balance(paper_wallet.balance)
        last_balance = notify_balance_change(
            notifier,
            last_balance,
            float(paper_wallet.balance),
            balance_updates,
        )

    monitor_task = asyncio.create_task(
        console_monitor.monitor_loop(
            exchange,
            paper_wallet,
            LOG_DIR / "bot.log",
            quiet_mode=config.get("quiet_mode", False),
        )
    )

    max_open_trades = config.get("max_open_trades", 1)
    position_guard = OpenPositionGuard(max_open_trades)
    rotator = PortfolioRotator()

    mode = user.get("mode", config.get("mode", "auto"))
    state = {"running": True, "mode": mode}
    # Caches for OHLCV and regime data are stored on the session_state
    session_state = SessionState(last_balance=last_balance)
    last_candle_ts: dict[str, int] = {}

    control_task = asyncio.create_task(console_control.control_loop(state))
    rotation_task = asyncio.create_task(
        _rotation_loop(
            rotator,
            exchange,
            user.get("wallet_address", ""),
            state,
            notifier,
            check_balance_change,
        )
    )
    solana_scan_task: asyncio.Task | None = None
    if config.get("solana_scanner", {}).get("enabled"):
        solana_scan_task = asyncio.create_task(solana_scan_loop())
    registry_task = asyncio.create_task(
        registry_update_loop(
            config.get("token_registry", {}).get("refresh_interval_minutes", 15)
        )
    )
    print("Bot running. Type 'stop' to pause, 'start' to resume, 'quit' to exit.")

    from crypto_bot.telegram_bot_ui import TelegramBotUI

    telegram_bot = (
        TelegramBotUI(
            notifier,
            state,
            LOG_DIR / "bot.log",
            rotator,
            exchange,
            user.get("wallet_address", ""),
            command_cooldown=config.get("telegram", {}).get("command_cooldown", 5),
        )
        if notifier.enabled and notifier.token and notifier.chat_id
        else None
    )

    if telegram_bot:
        telegram_bot.run_async()

    meme_wave_task = None
    if config.get("meme_wave_sniper", {}).get("enabled"):
        from crypto_bot.solana import start_runner

        meme_wave_task = start_runner(config.get("meme_wave_sniper", {}))
    sniper_cfg = config.get("meme_wave_sniper", {})
    sniper_task = None
    if sniper_cfg.get("enabled"):
        from crypto_bot.solana.runner import run as sniper_run

        sniper_task = asyncio.create_task(sniper_run(sniper_cfg))

    if config.get("scan_in_background", True):
        session_state.scan_task = asyncio.create_task(
            initial_scan(
                exchange,
                config,
                session_state,
                notifier if status_updates else None,
            )
        )
    else:
        await initial_scan(
            exchange,
            config,
            session_state,
            notifier if status_updates else None,
        )

    ctx = BotContext(
        positions=session_state.positions,
        df_cache=session_state.df_cache,
        regime_cache=session_state.regime_cache,
        config=config,
        mempool_monitor=mempool_monitor,
        mempool_cfg=mempool_cfg,
    )
    ctx.exchange = exchange
    ctx.secondary_exchange = secondary_exchange
    ctx.ws_client = ws_client
    ctx.risk_manager = risk_manager
    ctx.notifier = notifier
    ctx.paper_wallet = paper_wallet
    ctx.position_guard = position_guard
    ctx.balance = await fetch_and_log_balance(exchange, paper_wallet, config)
    last_balance = ctx.balance
    runner = PhaseRunner(
        [
            fetch_candidates,
            update_caches,
            enrich_with_pyth,
            analyse_batch,
            refresh_balance,
            execute_signals,
            handle_exits,
        ]
    )

    loop_count = 0
    last_weight_update = last_optimize = 0.0

    try:
        while True:
            maybe_reload_config(state, config)
            reload_config(
                config,
                ctx,
                risk_manager,
                rotator,
                position_guard,
                force=state.get("reload", False),
            )
            state["reload"] = False

            if state.get("liquidate_all"):
                await force_exit_all(ctx)
                state["liquidate_all"] = False

            if config.get("arbitrage_enabled", True):
                try:
                    if ctx.secondary_exchange:
                        arb_syms = await scan_cex_arbitrage(
                            exchange, ctx.secondary_exchange, config
                        )
                    else:
                        arb_syms = await scan_arbitrage(exchange, config)
                    if arb_syms:
                        async with QUEUE_LOCK:
                            for sym in reversed(arb_syms):
                                symbol_priority_queue.appendleft(sym)
                except Exception as exc:  # pragma: no cover - best effort
                    logger.error("Arbitrage scan error: %s", exc)

            cycle_start = time.perf_counter()
            ctx.timing = await runner.run(ctx)
            loop_count += 1

            if time.time() - last_weight_update >= 86400:
                weights = compute_strategy_weights()
                if weights:
                    logger.info("Updating strategy allocation to %s", weights)
                    risk_manager.update_allocation(weights)
                    config["strategy_allocation"] = weights
                last_weight_update = time.time()

            if config.get("optimization", {}).get("enabled"):
                if (
                    time.time() - last_optimize
                    >= config["optimization"].get("interval_days", 7) * 86400
                ):
                    optimize_strategies()
                    last_optimize = time.time()

            if not state.get("running"):
                await asyncio.sleep(1)
                continue

            balances = await asyncio.to_thread(
                check_wallet_balances, user.get("wallet_address", "")
            )
            quote_token = config.get("auto_convert_quote", "USDC")
            for token in detect_non_trade_tokens(balances):
                amount = balances[token]
                logger.info("Converting %s %s to %s", amount, token, quote_token)
                await auto_convert_funds(
                    user.get("wallet_address", ""),
                    token,
                    quote_token,
                    amount,
                    dry_run=config["execution_mode"] == "dry_run",
                    slippage_bps=config.get("solana_slippage_bps", 50),
                    notifier=notifier,
                    mempool_monitor=ctx.mempool_monitor,
                    mempool_cfg=ctx.mempool_cfg,
                )
                if asyncio.iscoroutinefunction(
                    getattr(exchange, "fetch_balance", None)
                ):
                    bal = await exchange.fetch_balance()
                else:
                    bal = await asyncio.to_thread(exchange.fetch_balance)
                bal_val = (
                    bal.get("USDT", {}).get("free", 0)
                    if isinstance(bal.get("USDT"), dict)
                    else bal.get("USDT", 0)
                )
                check_balance_change(float(bal_val), "funds converted")

            # Refresh OHLCV for open positions if a new candle has formed
            tf = config.get("timeframe", "1h")
            tf_sec = timeframe_seconds(None, tf)
            open_syms: list[str] = []
            for sym in ctx.positions:
                last_ts = last_candle_ts.get(sym, 0)
                if time.time() - last_ts >= tf_sec:
                    open_syms.append(sym)
            if open_syms:
                ohlcv_batch_size = config.get("ohlcv_batch_size")
                if ohlcv_batch_size is None:
                    ohlcv_batch_size = config.get("symbol_filter", {}).get(
                        "ohlcv_batch_size"
                    )
                async with OHLCV_LOCK:
                    tf_cache = ctx.df_cache.get(tf, {})
                    tf_cache = await update_ohlcv_cache(
                        exchange,
                        tf_cache,
                        open_syms,
                        timeframe=tf,
                        limit=2,
                        use_websocket=config.get("use_websocket", False),
                        force_websocket_history=config.get(
                            "force_websocket_history", False
                        ),
                        max_concurrent=config.get("max_concurrent_ohlcv"),
                        config=config,
                        batch_size=ohlcv_batch_size,
                    )
                    ctx.df_cache[tf] = tf_cache
                    session_state.df_cache[tf] = tf_cache
                for sym in open_syms:
                    df = tf_cache.get(sym)
                    if df is not None and not df.empty:
                        last_candle_ts[sym] = int(df["timestamp"].iloc[-1])
                        higher_df = ctx.df_cache.get(
                            config.get("higher_timeframe", "1d"), {}
                        ).get(sym)
                        regime, _ = await classify_regime_async(df, higher_df)
                        ctx.positions[sym]["regime"] = regime
                        TOTAL_ANALYSES += 1
                        if regime == "unknown":
                            UNKNOWN_COUNT += 1

            total_time = time.perf_counter() - cycle_start
            timing = getattr(ctx, "timing", {})
            _emit_timing(
                timing.get("fetch_candidates", 0.0),
                timing.get("update_caches", 0.0),
                timing.get("analyse_batch", 0.0),
                total_time,
                metrics_path,
                timing.get("ohlcv_fetch_latency", 0.0),
                timing.get("execution_latency", 0.0),
            )

            if config.get("metrics_enabled") and config.get("metrics_backend") == "csv":
                metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "ticker_fetch_time": timing.get("fetch_candidates", 0.0),
                    "symbol_filter_ratio": timing.get("symbol_filter_ratio", 1.0),
                    "ohlcv_fetch_latency": timing.get("ohlcv_fetch_latency", 0.0),
                    "execution_latency": timing.get("execution_latency", 0.0),
                    "unknown_regimes": sum(
                        1
                        for r in getattr(ctx, "analysis_results", [])
                        if r.get("regime") == "unknown"
                    ),
                }
                write_cycle_metrics(metrics, config)

            unknown_rate = UNKNOWN_COUNT / max(TOTAL_ANALYSES, 1)
            if unknown_rate > 0.2 and ctx.notifier:
                ctx.notifier.notify(f"⚠️ Unknown regime rate {unknown_rate:.1%}")
            delay = config.get("loop_interval_minutes", 1) / max(
                ctx.volatility_factor, 1e-6
            )
            logger.info("Sleeping for %.2f minutes", delay)
            await asyncio.sleep(delay * 60)

    finally:
        if hasattr(exchange, "close"):
            if asyncio.iscoroutinefunction(getattr(exchange, "close")):
                with contextlib.suppress(Exception):
                    await exchange.close()
            else:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(exchange.close)
        if solana_scan_task:
            solana_scan_task.cancel()
            try:
                await solana_scan_task
            except asyncio.CancelledError:
                pass
        if registry_task:
            registry_task.cancel()
            try:
                await registry_task
            except asyncio.CancelledError:
                pass
        if session_state.scan_task:
            session_state.scan_task.cancel()
            try:
                await session_state.scan_task
            except asyncio.CancelledError:
                pass
        monitor_task.cancel()
        if control_task:
            control_task.cancel()
        rotation_task.cancel()
        if sniper_task:
            sniper_task.cancel()
        if meme_wave_task:
            meme_wave_task.cancel()
            try:
                await meme_wave_task
            except asyncio.CancelledError:
                pass
        if telegram_bot:
            telegram_bot.stop()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        try:
            await rotation_task
        except asyncio.CancelledError:
            pass
        if sniper_task:
            try:
                await sniper_task
            except asyncio.CancelledError:
                pass
        if control_task:
            try:
                await control_task
            except asyncio.CancelledError:
                pass
        for task in list(WS_PING_TASKS):
            task.cancel()
        for task in list(WS_PING_TASKS):
            try:
                await task
            except asyncio.CancelledError:
                pass
        WS_PING_TASKS.clear()
        for task in list(SNIPER_TASKS):
            task.cancel()
        for task in list(SNIPER_TASKS):
            try:
                await task
            except asyncio.CancelledError:
                pass
        SNIPER_TASKS.clear()

    return notifier


async def main() -> None:
    """Entry point for running the trading bot with error handling."""
    notifier: TelegramNotifier | None = None
    try:
        from crypto_bot.utils.token_registry import refresh_mints

        await refresh_mints()
        notifier = await _main_impl()
    except Exception as exc:  # pragma: no cover - error path
        logger.exception("Unhandled error in main: %s", exc)
        if notifier:
            notifier.notify(f"❌ Bot stopped: {exc}")
    finally:
        if notifier:
            notifier.notify("Bot shutting down")
        logger.info("Bot shutting down")


if __name__ == "__main__":  # pragma: no cover - manual execution
    asyncio.run(main())
