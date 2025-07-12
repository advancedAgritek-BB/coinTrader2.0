import os
import asyncio
import contextlib
import json
import time
from pathlib import Path
from datetime import datetime
from collections import deque, OrderedDict, defaultdict
from cachetools import LRUCache
from dataclasses import dataclass, field

# Track WebSocket ping tasks
WS_PING_TASKS: set[asyncio.Task] = set()

try:
    import ccxt  # type: ignore
    from ccxt.base.errors import NetworkError, ExchangeError
except Exception:  # pragma: no cover - optional dependency
    import types

    ccxt = types.SimpleNamespace(NetworkError=Exception, ExchangeError=Exception)

import pandas as pd
import yaml
from dotenv import dotenv_values
from pydantic import ValidationError

from schema.scanner import ScannerConfig

from crypto_bot.utils.telegram import TelegramNotifier, send_test_message
from crypto_bot.utils.trade_reporter import report_entry, report_exit
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.portfolio_rotator import PortfolioRotator
from crypto_bot.wallet_manager import load_or_create
from crypto_bot.utils.market_analyzer import analyze_symbol
from crypto_bot.strategy_router import strategy_for, strategy_name
from crypto_bot.cooldown_manager import (
    configure as cooldown_configure,
    in_cooldown,
    mark_cooldown,
)
from crypto_bot.phase_runner import BotContext, PhaseRunner
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
from crypto_bot.risk.exit_manager import (
    calculate_trailing_stop,
    should_exit,
    get_partial_exit_percent,
)
from crypto_bot.execution.cex_executor import (
    execute_trade_async as cex_trade_async,
    get_exchange,
    place_stop_order,
)
from crypto_bot.open_position_guard import OpenPositionGuard
from crypto_bot import console_monitor, console_control
from crypto_bot.utils.performance_logger import log_performance
from crypto_bot.utils.position_logger import log_position, log_balance
from crypto_bot.utils.regime_logger import log_regime
from crypto_bot.utils.market_loader import (
    load_kraken_symbols,
    update_ohlcv_cache,
    update_multi_tf_ohlcv_cache,
    update_regime_tf_cache,
    timeframe_seconds,
    configure as market_loader_configure,
)
from crypto_bot.utils.eval_queue import build_priority_queue
from crypto_bot.utils.symbol_utils import get_filtered_symbols, fix_symbol
from crypto_bot.utils.metrics_logger import log_cycle as log_cycle_metrics
from crypto_bot.utils.pnl_logger import log_pnl
from crypto_bot.utils.regime_pnl_tracker import log_trade as log_regime_pnl
from crypto_bot.utils.regime_pnl_tracker import get_recent_win_rate
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.utils.strategy_utils import compute_strategy_weights
from crypto_bot.auto_optimizer import optimize_strategies
from crypto_bot.utils.telemetry import telemetry, write_cycle_metrics
from crypto_bot.utils.correlation import compute_correlation_matrix
from crypto_bot.utils.strategy_analytics import write_scores, write_stats
from crypto_bot.utils.balance import get_usdt_balance
from crypto_bot.fund_manager import (
    auto_convert_funds,
    check_wallet_balances,
    detect_non_trade_tokens,
)
from crypto_bot.regime.regime_classifier import CONFIG


def _fix_symbol(sym: str) -> str:
    """Internal wrapper for tests to normalize symbols."""
    return fix_symbol(sym)


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
ENV_PATH = Path(__file__).resolve().parent / ".env"

logger = setup_logger("bot", LOG_DIR / "bot.log", to_console=False)

# Queue of symbols awaiting evaluation across loops
symbol_priority_queue: deque[str] = deque()


# Protects shared queues for future multi-tasking scenarios
QUEUE_LOCK = asyncio.Lock()

# Retry parameters for the initial symbol scan
MAX_SYMBOL_SCAN_ATTEMPTS = 3
SYMBOL_SCAN_RETRY_DELAY = 10
MAX_SYMBOL_SCAN_DELAY = 60

# Maximum number of symbols per timeframe to keep in the OHLCV cache
DF_CACHE_MAX_SIZE = 500


@dataclass
class SessionState:
    """Runtime session state shared across tasks."""

    positions: dict[str, dict] = field(default_factory=dict)
    df_cache: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    global_cache_limit: int = 1000
    ohlcv_cache: LRUCache = field(init=False)
    regime_cache: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    last_balance: float | None = None
    scan_task: asyncio.Task | None = None

    def __post_init__(self) -> None:
        self.ohlcv_cache = LRUCache(self.global_cache_limit)


def update_df_cache(
    cache: dict[str, dict[str, pd.DataFrame]],
    timeframe: str,
    symbol: str,
    df: pd.DataFrame,
    max_size: int = DF_CACHE_MAX_SIZE,
    global_cache: LRUCache | None = None,
) -> None:
    """Update an OHLCV cache with LRU eviction."""
    tf_cache = cache.setdefault(timeframe, OrderedDict())
    tf_cache[symbol] = df
    tf_cache.move_to_end(symbol)
    if len(tf_cache) > max_size:
        old_sym, _ = tf_cache.popitem(last=False)
        if global_cache is not None:
            global_cache.pop((timeframe, old_sym), None)
    if global_cache is not None:
        global_cache[(timeframe, symbol)] = df


def direction_to_side(direction: str) -> str:
    """Translate strategy direction to trade side."""
    return "buy" if direction == "long" else "sell"


def opposite_side(side: str) -> str:
    """Return the opposite trading side."""
    return "sell" if side == "buy" else "buy"


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
        return await get_usdt_balance(exchange, config)
    return paper_wallet.balance if paper_wallet else 0.0


async def fetch_and_log_balance(exchange, paper_wallet, config):
    """Return the latest wallet balance and log it."""
    latest_balance = await fetch_balance(exchange, paper_wallet, config)
    log_balance(float(latest_balance))
    return latest_balance


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

    if "symbol" in data:
        data["symbol"] = fix_symbol(data["symbol"])
    if "symbols" in data:
        data["symbols"] = [fix_symbol(s) for s in data.get("symbols", [])]
    try:
        if hasattr(ScannerConfig, "model_validate"):
            ScannerConfig.model_validate(data)
        else:  # pragma: no cover - for Pydantic < 2
            ScannerConfig.parse_obj(data)
    except ValidationError as exc:
        print("Invalid configuration:\n", exc)
        raise SystemExit(1)
    return data


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


def _cast_to_type(value: str, example: object) -> object:
    """Try to cast a string to the type of `example`."""
    target_type = type(example)
    if target_type is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    try:
        return target_type(value)
    except Exception:
        return value


async def _ws_ping_loop(exchange: ccxt.Exchange, interval: float) -> None:
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
                        for client in clients.values():
                            if is_coro:
                                await ping(client)
                            else:
                                await asyncio.to_thread(ping, client)
                    else:
                        continue
                else:
                    if is_coro:
                        await ping()
                    else:
                        await asyncio.to_thread(ping)
            except asyncio.CancelledError:
                raise
            except ccxt.NetworkError as exc:
                logger.warning("Network error: %s; retrying shortly", exc)
                await asyncio.sleep(5)
                continue
            except ccxt.ExchangeError as exc:
                logger.error("Exchange error: %s", exc)
                await asyncio.sleep(5)
                continue
            except Exception as exc:  # pragma: no cover - ping failures
                logger.error("WebSocket ping failed: %s", exc, exc_info=True)
    except asyncio.CancelledError:
        pass




async def initial_scan(
    exchange: ccxt.Exchange,
    config: dict,
    state: SessionState,
    notifier: TelegramNotifier | None = None,
) -> None:
    """Populate OHLCV and regime caches before trading begins."""

    symbols = config.get("symbols") or [config.get("symbol")]
    if not symbols:
        return

    batch_size = int(config.get("symbol_batch_size", 10))
    total = len(symbols)
    processed = 0

    for i in range(0, total, batch_size):
        batch = symbols[i : i + batch_size]

        state.df_cache = await update_multi_tf_ohlcv_cache(
            exchange,
            state.df_cache,
            batch,
            config,
            limit=config.get("scan_lookback_limit", 50),
            timeframe_limits=config.get("timeframe_limits"),
            use_websocket=config.get("use_websocket", False),
            force_websocket_history=config.get("force_websocket_history", False),
            max_concurrent=config.get("max_concurrent_ohlcv"),
            notifier=notifier,
        )

        state.regime_cache = await update_regime_tf_cache(
            exchange,
            state.regime_cache,
            batch,
            config,
            limit=config.get("scan_lookback_limit", 50),
            timeframe_limits=config.get("timeframe_limits"),
            use_websocket=config.get("use_websocket", False),
            force_websocket_history=config.get("force_websocket_history", False),
            max_concurrent=config.get("max_concurrent_ohlcv"),
            notifier=notifier,
            df_map=state.df_cache,
        )

        processed += len(batch)
        pct = processed / total * 100
        logger.info("Initial scan %.1f%% complete", pct)
        if notifier and config.get("telegram", {}).get("status_updates", True):
            notifier.notify(f"Initial scan {pct:.1f}% complete")

    return


async def fetch_candidates(ctx: BotContext) -> None:
    """Gather symbols for this cycle and build the evaluation batch."""
    t0 = time.perf_counter()
    symbols = await get_filtered_symbols(ctx.exchange, ctx.config)
    ctx.timing["symbol_time"] = time.perf_counter() - t0

    total_available = len(ctx.config.get("symbols") or [ctx.config.get("symbol")])
    ctx.timing["symbol_filter_ratio"] = (
        len(symbols) / total_available if total_available else 1.0
    )

    global symbol_priority_queue
    batch_size = ctx.config.get("symbol_batch_size", 10)
    async with QUEUE_LOCK:
        if not symbol_priority_queue:
            symbol_priority_queue = build_priority_queue(symbols)
        if len(symbol_priority_queue) < batch_size:
            symbol_priority_queue.extend(build_priority_queue(symbols))
        ctx.current_batch = [
            symbol_priority_queue.popleft()
            for _ in range(min(batch_size, len(symbol_priority_queue)))
        ]


async def update_caches(ctx: BotContext) -> None:
    """Update OHLCV and regime caches for the current symbol batch."""
    batch = ctx.current_batch
    if not batch:
        return

    start = time.perf_counter()
    tf_minutes = int(pd.Timedelta(ctx.config.get("timeframe", "1h")).total_seconds() // 60)
    limit = min(150, tf_minutes * 2)
    limit = int(ctx.config.get("cycle_lookback_limit") or limit)

    ctx.df_cache = await update_multi_tf_ohlcv_cache(
        ctx.exchange,
        ctx.df_cache,
        batch,
        ctx.config,
        limit=limit,
        timeframe_limits=ctx.config.get("timeframe_limits"),
        use_websocket=ctx.config.get("use_websocket", False),
        force_websocket_history=ctx.config.get("force_websocket_history", False),
        max_concurrent=ctx.config.get("max_concurrent_ohlcv"),
        notifier=ctx.notifier if ctx.config.get("telegram", {}).get("status_updates", True) else None,
    )

    ctx.regime_cache = await update_regime_tf_cache(
        ctx.exchange,
        ctx.regime_cache,
        batch,
        ctx.config,
        limit=limit,
        timeframe_limits=ctx.config.get("timeframe_limits"),
        use_websocket=ctx.config.get("use_websocket", False),
        force_websocket_history=ctx.config.get("force_websocket_history", False),
        max_concurrent=ctx.config.get("max_concurrent_ohlcv"),
        notifier=ctx.notifier if ctx.config.get("telegram", {}).get("status_updates", True) else None,
        df_map=ctx.df_cache,
    )

    ctx.timing["ohlcv_fetch_latency"] = time.perf_counter() - start


async def analyse_batch(ctx: BotContext) -> None:
    """Run signal analysis on the current batch."""
    batch = ctx.current_batch
    if not batch:
        ctx.analysis_results = []
        return

    tasks = []
    mode = ctx.config.get("mode", "cex")
    for sym in batch:
        df_map = {tf: c.get(sym) for tf, c in ctx.df_cache.items()}
        for tf, cache in ctx.regime_cache.items():
            df_map[tf] = cache.get(sym)
        tasks.append(analyze_symbol(sym, df_map, mode, ctx.config, ctx.notifier))

    ctx.analysis_results = await asyncio.gather(*tasks)


async def execute_signals(ctx: BotContext) -> None:
    """Open trades for qualified analysis results."""
    results = getattr(ctx, "analysis_results", [])
    if not results:
        return

    # Prioritize by score
    results = [r for r in results if not r.get("skip") and r.get("direction") != "none"]
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_n = ctx.config.get("top_n_symbols", 3)

    for candidate in results[:top_n]:
        if not ctx.position_guard or not ctx.position_guard.can_open(ctx.positions):
            break
        sym = candidate["symbol"]
        if sym in ctx.positions:
            continue

        df = candidate["df"]
        price = df["close"].iloc[-1]
        score = candidate.get("score", 0.0)
        strategy = candidate.get("name", "")
        allowed, _ = ctx.risk_manager.allow_trade(df, strategy)
        if not allowed:
            continue

        size = ctx.risk_manager.position_size(
            score,
            ctx.balance,
            df,
            atr=candidate.get("atr"),
            price=price,
        )

        if not ctx.risk_manager.can_allocate(strategy, size, ctx.balance):
            continue

        if price <= 0:
            logger.warning("Price for %s is %s; skipping trade", sym, price)
            continue
        amount = size / price
        start_exec = time.perf_counter()
        await cex_trade_async(
            ctx.exchange,
            ctx.ws_client,
            sym,
            direction_to_side(candidate["direction"]),
            amount,
            ctx.notifier,
            dry_run=ctx.config.get("execution_mode") == "dry_run",
            use_websocket=ctx.config.get("use_websocket", False),
            config=ctx.config,
        )
        ctx.timing["execution_latency"] = max(
            ctx.timing.get("execution_latency", 0.0),
            time.perf_counter() - start_exec,
        )

        ctx.risk_manager.allocate_capital(strategy, size)
        if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
            ctx.paper_wallet.open(sym, direction_to_side(candidate["direction"]), amount, price)
            ctx.balance = ctx.paper_wallet.balance
        ctx.positions[sym] = {
            "side": direction_to_side(candidate["direction"]),
            "entry_price": price,
            "entry_time": datetime.utcnow().isoformat(),
            "regime": candidate.get("regime"),
            "strategy": strategy,
            "confidence": score,
            "pnl": 0.0,
            "size": amount,
            "trailing_stop": 0.0,
            "highest_price": price,
        }
        try:
            log_position(
                sym,
                direction_to_side(candidate["direction"]),
                amount,
                price,
                price,
                ctx.balance,
            )
        except Exception:
            pass


async def handle_exits(ctx: BotContext) -> None:
    """Check open positions for exit conditions."""
    tf = ctx.config.get("timeframe", "1h")
    tf_cache = ctx.df_cache.get(tf, {})
    for sym, pos in list(ctx.positions.items()):
        df = tf_cache.get(sym)
        if df is None or df.empty:
            continue
        current_price = float(df["close"].iloc[-1])
        # "buy" positions are treated as long when computing PnL
        pnl_pct = (
            (current_price - pos["entry_price"]) / pos["entry_price"]
        ) * (1 if pos["side"] == "buy" else -1)
        if pnl_pct >= ctx.config.get("exit_strategy", {}).get("min_gain_to_trail", 0):
            if current_price > pos.get("highest_price", pos["entry_price"]):
                pos["highest_price"] = current_price
            pos["trailing_stop"] = calculate_trailing_stop(
                pd.Series([pos.get("highest_price", current_price)]),
                ctx.config.get("exit_strategy", {}).get("trailing_stop_pct", 0.1),
            )
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


async def _rotation_loop(
    rotator: PortfolioRotator,
    exchange: ccxt.Exchange,
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
                current_balance = await get_usdt_balance(exchange, rotator.config)
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
                    bal = await exchange.fetch_balance()
                else:
                    bal = await asyncio.to_thread(exchange.fetch_balance)
                check_balance_change(float(current_balance), "external change")
                holdings = {
                    k: (v.get("total") if isinstance(v, dict) else v)
                    for k, v in bal.items()
                }
                await rotator.rotate(exchange, wallet, holdings, notifier)
        except asyncio.CancelledError:
            break
        except ccxt.NetworkError as exc:
            logger.warning("Network error: %s; retrying shortly", exc)
            await asyncio.sleep(5)
            continue
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error: %s", exc)
            await asyncio.sleep(5)
            continue
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
    config = load_config()
    dotenv_values(ENV_PATH)
    user = load_or_create()
    tg_cfg = {
        "token": user.get("telegram_token", ""),
        "chat_id": user.get("telegram_chat_id", ""),
        "enabled": True,
    }
    notifier = TelegramNotifier.from_config(tg_cfg)
    send_test_message(notifier.token, notifier.chat_id, "Bot started")
    get_exchange(config)
    return notifier
async def main() -> None:
    """Entry point for running the trading bot with error handling."""
    notifier: TelegramNotifier | None = None
    try:
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
