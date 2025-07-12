import os
import asyncio
import contextlib
import json
import time
from pathlib import Path
from datetime import datetime
from collections import deque, OrderedDict, defaultdict
from dataclasses import dataclass, field

# Track WebSocket ping tasks
WS_PING_TASKS: set[asyncio.Task] = set()

try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    import types

    ccxt = types.SimpleNamespace()

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
from crypto_bot.strategy.grid_bot import GridConfig
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
from crypto_bot import grid_state
from crypto_bot.utils.strategy_utils import compute_strategy_weights
from crypto_bot.auto_optimizer import optimize_strategies
from crypto_bot.utils.telemetry import telemetry, write_cycle_metrics
from crypto_bot.utils.correlation import compute_correlation_matrix
from crypto_bot.utils.strategy_analytics import write_scores, write_stats
from crypto_bot.fund_manager import (
    auto_convert_funds,
    check_wallet_balances,
    detect_non_trade_tokens,
)
from crypto_bot.utils.balance import get_usdt_balance, get_btc_balance
from crypto_bot.regime.regime_classifier import CONFIG


def _fix_symbol(sym: str) -> str:
    """Internal wrapper for tests to normalize symbols."""
    return fix_symbol(sym)


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
ENV_PATH = Path(__file__).resolve().parent / ".env"

logger = setup_logger("bot", LOG_DIR / "bot.log", to_console=False)

# Queue of symbols awaiting evaluation across loops
symbol_priority_queue: deque[str] = deque()

# Queue tracking symbols evaluated across cycles
SYMBOL_EVAL_QUEUE: deque[str] = deque()

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
            except Exception as exc:  # pragma: no cover - ping failures
                logger.error("WebSocket ping failed: %s", exc, exc_info=True)
    except asyncio.CancelledError:
        pass




async def initial_scan(
    exchange: object,
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

    mode = ctx.config.get("mode", "cex")
    sem = asyncio.Semaphore(ctx.config.get("analysis_concurrency", 10))

    async def bounded_analyze(sym: str):
        async with sem:
            df_map = {tf: c.get(sym) for tf, c in ctx.df_cache.items()}
            for tf, cache in ctx.regime_cache.items():
                df_map[tf] = cache.get(sym)
            return await analyze_symbol(sym, df_map, mode, ctx.config, ctx.notifier)

    tasks = [bounded_analyze(sym) for sym in batch]
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

        leverage = candidate.get("leverage", 1)
        size *= leverage

        if not ctx.risk_manager.can_allocate(strategy, size, ctx.balance):
            continue

        amount = size / price if price > 0 else 0.0
        start_exec = time.perf_counter()
        params = None
        if candidate.get("name") == "grid_bot":
            grid_cfg = GridConfig.from_dict(ctx.config.get("grid_bot"))
            params = {"leverage": grid_cfg.leverage}
        if "exchange_buy" in candidate and "exchange_sell" in candidate:
            await cex_trade_async(
                ctx.exchange,
                ctx.ws_client,
                sym,
                "buy",
                amount,
                ctx.notifier,
                dry_run=ctx.config.get("execution_mode") == "dry_run",
                use_websocket=ctx.config.get("use_websocket", False),
                config=ctx.config,
                params=params,
                leverage=leverage,
                exchange_override=candidate["exchange_buy"],
            )
            await cex_trade_async(
                ctx.exchange,
                ctx.ws_client,
                sym,
                "sell",
                amount,
                ctx.notifier,
                dry_run=ctx.config.get("execution_mode") == "dry_run",
                use_websocket=ctx.config.get("use_websocket", False),
                config=ctx.config,
                params=params,
                leverage=leverage,
                exchange_override=candidate["exchange_sell"],
            )
        else:
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
                params=params,
                leverage=leverage,
            )
        ctx.timing["execution_latency"] = max(
            ctx.timing.get("execution_latency", 0.0),
            time.perf_counter() - start_exec,
        )

        ctx.risk_manager.allocate_capital(strategy, size)
        if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
            ctx.paper_wallet.open(sym, direction_to_side(candidate["direction"]), amount, price)
            ctx.balance = ctx.paper_wallet.balance
        ex_fields: dict[str, object] = {}
        if "exchange_buy" in candidate and "exchange_sell" in candidate:
            ex_fields["exchange_buy"] = candidate["exchange_buy"]
            ex_fields["exchange_sell"] = candidate["exchange_sell"]

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
            **ex_fields,
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
            profit = (current_price - pos["entry_price"]) * pos["size"]
            if pos["side"] == "sell":
                profit = -profit
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
            if profit > 0 and getattr(ctx, "user_wallet", None):
                await auto_convert_funds(
                    ctx.user_wallet,
                    "USDT",
                    "BTC",
                    profit,
                    dry_run=ctx.config.get("execution_mode") == "dry_run",
                    slippage_bps=ctx.config.get("solana_slippage_bps", 50),
                    notifier=ctx.notifier,
                )
                try:
                    tf_price = ctx.df_cache.get(tf, {}).get("BTC/USDT") or ctx.df_cache.get(tf, {}).get("XBT/USDT")
                    if tf_price is not None and not tf_price.empty:
                        btc_price = float(tf_price["close"].iloc[-1])
                        btc_amt = profit / btc_price
                        if ctx.notifier:
                            await ctx.notifier.send_message(f"Added {btc_amt:.6f} BTC to wallet")
                except Exception:
                    pass
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
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
                    bal = await exchange.fetch_balance()
                else:
                    bal = await asyncio.to_thread(exchange.fetch_balance)
                current_balance = (
                    bal.get("USDT", {}).get("free", 0)
                    if isinstance(bal.get("USDT"), dict)
                    else bal.get("USDT", 0)
                )
                check_balance_change(
                    "USDT", float(current_balance), "external change"
                )
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


async def handle_fund_conversions(
    exchange: object,
    config: dict,
    notifier: TelegramNotifier | None,
    user_wallet: str,
    check_balance_change: callable,
) -> None:
    """Convert unsupported funding tokens to BTC and update balance."""

    balances = await asyncio.to_thread(check_wallet_balances, user_wallet)
    for token in detect_non_trade_tokens(balances):
        amount = balances[token]
        logger.info("Converting %s %s to BTC", amount, token)
        await auto_convert_funds(
            user_wallet,
            token,
            "BTC",
            amount,
            dry_run=config.get("execution_mode") == "dry_run",
            slippage_bps=config.get("solana_slippage_bps", 50),
            notifier=notifier,
        )
        bal_val = await get_btc_balance(exchange, config)
        check_balance_change(float(bal_val), "funds converted", currency="BTC")
        if notifier:
            notifier.notify(f"Converted to {bal_val:.6f} BTC")
        bal_val = await get_usdt_balance(exchange, config)
        check_balance_change("BTC", float(bal_val), "funds converted")


async def refresh_open_position_data(
    exchange: object,
    ctx: BotContext,
    last_candle_ts: dict[str, int],
    config: dict,
) -> None:
    """Update OHLCV data for open positions if a new candle formed."""

    tf = config.get("timeframe", "1h")
    tf_sec = timeframe_seconds(None, tf)
    open_syms: list[str] = []
    for sym in ctx.positions:
        last_ts = last_candle_ts.get(sym, 0)
        if time.time() - last_ts >= tf_sec:
            open_syms.append(sym)
    if open_syms:
        tf_cache = ctx.df_cache.get(tf, {})
        tf_cache = await update_ohlcv_cache(
            exchange,
            tf_cache,
            open_syms,
            timeframe=tf,
            limit=2,
            use_websocket=config.get("use_websocket", False),
            force_websocket_history=config.get("force_websocket_history", False),
            max_concurrent=config.get("max_concurrent_ohlcv"),
        )
        ctx.df_cache[tf] = tf_cache
        for sym in open_syms:
            df = tf_cache.get(sym)
            if df is not None and not df.empty:
                last_candle_ts[sym] = int(df["timestamp"].iloc[-1])


async def _price_model_training_loop(ctx: BotContext) -> None:
    """Periodically train the torch price model."""
    interval = (
        ctx.config.get("torch_price_model", {}).get("training_interval_hours", 24)
        * 3600
    )
    while True:
        try:
            from crypto_bot import torch_price_model

            torch_price_model.train_model(ctx.df_cache)
        except asyncio.CancelledError:
            break
        except Exception as exc:  # pragma: no cover - training errors
            logger.error("Price model training error: %s", exc, exc_info=True)
        sleep_remaining = interval
        while sleep_remaining > 0:
            sleep_chunk = min(60, sleep_remaining)
            try:
                await asyncio.sleep(sleep_chunk)
            except asyncio.CancelledError:
                return
            sleep_remaining -= sleep_chunk


async def _main_impl() -> TelegramNotifier:
    """Implementation for running the trading bot."""

    logger.info("Starting bot")
    config = load_config()
    dotenv_values(ENV_PATH)
    user = load_or_create()
    status_updates = config.get("telegram", {}).get("status_updates", True)
    balance_updates = config.get("telegram", {}).get("balance_updates", False)
    volume_ratio = config.get("volume_ratio", 1.0)
    tg_cfg = {
        "token": user.get("telegram_token", ""),
        "chat_id": user.get("telegram_chat_id", ""),
        "enabled": True,
    }
    notifier = TelegramNotifier.from_config(tg_cfg)
    send_test_message(notifier.token, notifier.chat_id, "Bot started")
    get_exchange(config)
    if status_updates:
        notifier.notify("🤖 CoinTrader2.0 started")

    if notifier.token and notifier.chat_id:
        if not send_test_message(notifier.token, notifier.chat_id, "Bot started"):
            logger.warning("Telegram test message failed; check your token and chat ID")

    # allow user-configured exchange to override YAML setting
    if user.get("exchange"):
        config["exchange"] = user["exchange"]

    exchange, ws_client = get_exchange(config)

    if not hasattr(exchange, "load_markets"):
        logger.error(
            "The installed ccxt package is missing or a local stub is in use."
        )
        if status_updates:
            notifier.notify(
                "❌ ccxt library not found or stubbed; check your installation"
            )
        return notifier

    if config.get("scan_markets", False) and not config.get("symbols"):
        attempt = 0
        delay = SYMBOL_SCAN_RETRY_DELAY
        discovered: list[str] | None = None
        while attempt < MAX_SYMBOL_SCAN_ATTEMPTS:
            discovered = await load_kraken_symbols(
                exchange,
                config.get("excluded_symbols", []),
                config,
            )
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
            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_SYMBOL_SCAN_DELAY)

        if discovered:
            config["symbols"] = discovered
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
    previous_balance: dict[str, float] = {"USDT": 0.0}

    def check_balance_change(
        new_balance: float,
        reason: str,
        *,
        currency: str = "USDT",
    ) -> None:
    previous_balance: dict[str, float] = {}

    def check_balance_change(currency: str, new_balance: float, reason: str) -> None:
        nonlocal previous_balance
        prev = previous_balance.get(currency, 0.0)
        delta = new_balance - prev
        if abs(delta) > balance_threshold and notifier:
            notifier.notify(
                f"Balance changed by {delta:.4f} {currency} due to {reason}"
            )
        previous_balance[currency] = new_balance

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
        previous_balance = {"USDT": float(init_bal)}
        previous_balance["USDT"] = float(init_bal)
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
        "volume_threshold_ratio", 0.1
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
        paper_wallet = PaperWallet(start_bal, config.get("max_open_trades", 1))
        log_balance(paper_wallet.balance)
        last_balance = notify_balance_change(
            notifier,
            last_balance,
            float(paper_wallet.balance),
            balance_updates,
        )

    monitor_task = asyncio.create_task(
        console_monitor.monitor_loop(exchange, paper_wallet, LOG_DIR / "bot.log")
    )

    position_tasks: dict[str, asyncio.Task] = {}
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
        if notifier.enabled
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
    )
    ctx.exchange = exchange
    ctx.ws_client = ws_client
    ctx.risk_manager = risk_manager
    ctx.notifier = notifier
    ctx.paper_wallet = paper_wallet
    ctx.position_guard = position_guard
    ctx.balance = last_balance
    runner = PhaseRunner([
        fetch_candidates,
        update_caches,
        analyse_batch,
        execute_signals,
        handle_exits,
    ])

    price_train_task = None
    if config.get("torch_price_model", {}).get("enabled"):
        price_train_task = asyncio.create_task(_price_model_training_loop(ctx))

    
    loop_count = 0
    last_weight_update = last_optimize = 0.0
    metrics_path = Path(config.get("metrics_output_file", LOG_DIR / "metrics.csv"))

    try:
        while True:
            delay_override: float | None = None
            try:
                cycle_start = time.perf_counter()
                ctx.timing = await runner.run(ctx)
                loop_count += 1

                if time.time() - last_weight_update >= config.get("weight_update_minutes", 60) * 60:
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
            except asyncio.CancelledError:
                raise
            except ccxt.NetworkError as exc:
                logger.warning("Network error: %s; retrying shortly", exc)
                delay_override = 5
            except ccxt.ExchangeError as exc:
                logger.error("Exchange error: %s", exc)
                delay_override = 5
            except Exception as exc:  # pragma: no cover - loop errors
                logger.error("Main loop error: %s", exc, exc_info=True)

            try:
                await handle_fund_conversions(
                    exchange,
                    config,
                    notifier,
                    user.get("wallet_address", ""),
                    check_balance_change,
                )

                await refresh_open_position_data(exchange, ctx, last_candle_ts, config)
            except Exception as exc:
                logger.error("Post-iteration error: %s", exc, exc_info=True)

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
                        1 for r in getattr(ctx, "analysis_results", []) if r.get("regime") == "unknown"
                    ),
                }
                write_cycle_metrics(metrics, config)
            logger.info("Sleeping for %s minutes", config["loop_interval_minutes"])
            delay = delay_override if delay_override is not None else pd.Timedelta(
                config["loop_interval_minutes"], unit="m"
            ).total_seconds()
            await asyncio.sleep(delay)
    finally:
        if session_state.scan_task:
            session_state.scan_task.cancel()
            try:
                await session_state.scan_task
            except asyncio.CancelledError:
                pass
        monitor_task.cancel()
        control_task.cancel()
        rotation_task.cancel()
        if sniper_task:
            sniper_task.cancel()
        if price_train_task:
            price_train_task.cancel()
        for task in list(position_tasks.values()):
            task.cancel()
        for task in list(position_tasks.values()):
            try:
                await task
            except asyncio.CancelledError:
                pass
        position_tasks.clear()
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
        if price_train_task:
            try:
                await price_train_task
            except asyncio.CancelledError:
                pass
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
        if hasattr(exchange, "close"):
            if asyncio.iscoroutinefunction(getattr(exchange, "close")):
                with contextlib.suppress(Exception):
                    await exchange.close()
            else:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(exchange.close)

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
