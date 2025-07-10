import os
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field

# Track WebSocket ping tasks
WS_PING_TASKS: set[asyncio.Task] = set()

import ccxt

import pandas as pd
import yaml
from dotenv import dotenv_values
from pydantic import ValidationError

from schema.scanner import ScannerConfig

from crypto_bot.utils.telegram import TelegramNotifier, send_test_message
from crypto_bot.utils.trade_reporter import report_entry, report_exit
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.portfolio_rotator import PortfolioRotator
from crypto_bot.auto_optimizer import optimize_strategies
from crypto_bot.wallet_manager import load_or_create
from crypto_bot.utils.market_analyzer import analyze_symbol
from crypto_bot.strategy_router import strategy_for, strategy_name
from crypto_bot.cooldown_manager import (
    in_cooldown,
    mark_cooldown,
    configure as cooldown_configure,
)
from crypto_bot.phase_runner import BotContext, PhaseRunner
from crypto_bot import grid_state
from crypto_bot.signals.signal_scoring import evaluate_async
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
from crypto_bot.execution.solana_executor import execute_swap
from crypto_bot.fund_manager import (
    check_wallet_balances,
    detect_non_trade_tokens,
    auto_convert_funds,
)
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.open_position_guard import OpenPositionGuard
from crypto_bot import console_monitor, console_control
from crypto_bot.utils.performance_logger import log_performance
from crypto_bot.utils.strategy_utils import compute_strategy_weights
from crypto_bot.utils.position_logger import log_position, log_balance
from crypto_bot.utils.regime_logger import log_regime
from crypto_bot.utils.metrics_logger import log_cycle as log_cycle_metrics
from crypto_bot.utils.market_loader import (
    load_kraken_symbols,
    load_ohlcv_parallel,
    update_ohlcv_cache,
    update_multi_tf_ohlcv_cache,
    update_regime_tf_cache,
    fetch_ohlcv_async,
    configure as market_loader_configure,
)
from crypto_bot.utils.eval_queue import build_priority_queue
from crypto_bot.utils.symbol_pre_filter import filter_symbols
from crypto_bot.utils.symbol_utils import get_filtered_symbols
from crypto_bot.utils.pnl_logger import log_pnl
from crypto_bot.utils.strategy_analytics import write_scores, write_stats
from crypto_bot.utils.regime_pnl_tracker import log_trade as log_regime_pnl
from crypto_bot.utils.regime_pnl_tracker import get_recent_win_rate
from crypto_bot.utils.trend_confirmation import confirm_multi_tf_trend
from crypto_bot.utils.correlation import compute_correlation_matrix
from crypto_bot.regime.regime_classifier import CONFIG
from crypto_bot.utils.metrics_logger import log_metrics_to_csv
from crypto_bot.utils.telemetry import telemetry



CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
ENV_PATH = Path(__file__).resolve().parent / ".env"

logger = setup_logger("bot", LOG_DIR / "bot.log", to_console=False)

# Queue of symbols awaiting evaluation across loops
symbol_priority_queue: deque[str] = deque()

# Queue tracking symbols evaluated across cycles
SYMBOL_EVAL_QUEUE: deque[str] = deque()

# Retry parameters for the initial symbol scan
MAX_SYMBOL_SCAN_ATTEMPTS = 3
SYMBOL_SCAN_RETRY_DELAY = 10
MAX_SYMBOL_SCAN_DELAY = 60


@dataclass
class SessionState:
    """Runtime session state shared across tasks."""

    positions: dict[str, dict] = field(default_factory=dict)
    df_cache: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    regime_cache: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    last_balance: float | None = None


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


def maybe_update_mode(
    state: dict,
    base_mode: str,
    config: dict,
    notifier: TelegramNotifier | None = None,
) -> None:
    """Switch bot mode based on recent win rate."""

    window = int(config.get("mode_degrade_window", 20))
    threshold = float(config.get("mode_threshold", 0.0))
    conservative = config.get("conservative_mode", "cex")

    win_rate = get_recent_win_rate(window)
    if win_rate < threshold:
        if state.get("mode") != conservative:
            state["mode"] = conservative
            if notifier:
                notifier.notify(
                    f"Win rate {win_rate:.2f} below {threshold:.2f}; mode set to {conservative}"
                )
    else:
        if state.get("mode") != base_mode:
            state["mode"] = base_mode
            if notifier:
                notifier.notify(f"Win rate recovered; mode set to {base_mode}")


def load_config() -> dict:
    """Load YAML configuration for the bot."""
    with open(CONFIG_PATH) as f:
        logger.info("Loading config from %s", CONFIG_PATH)
        data = yaml.safe_load(f) or {}
    try:
        ScannerConfig.model_validate(data)
    except ValidationError as exc:
        print("Invalid configuration:\n", exc)
        raise SystemExit(1)
    return data


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


async def _watch_position(
    symbol: str,
    exchange: object,
    state: SessionState,
    paper_wallet: PaperWallet | None,
    config: dict,
) -> None:
    """Live update position PnL and log changes."""
    use_ws = config.get("use_websocket", False) and hasattr(exchange, "watch_ticker")
    poll_interval = float(config.get("position_poll_interval", 5))
    ws_ping_interval = float(config.get("ws_ping_interval", 20))
    ping_task: asyncio.Task | None = None
    reconnect_attempts = 0
    max_reconnect = int(config.get("ws_max_retries", 3))

    while symbol in state.positions:
        try:
            if use_ws:
                if ping_task is None:
                    ping_task = asyncio.create_task(
                        _ws_ping_loop(exchange, ws_ping_interval)
                    )
                    WS_PING_TASKS.add(ping_task)
                try:
                    ticker = await asyncio.wait_for(
                        exchange.watch_ticker(symbol), timeout=5
                    )
                    reconnect_attempts = 0
                except asyncio.TimeoutError as to_exc:
                    if to_exc.args:
                        logger.warning(
                            "WebSocket ticker timeout for %s: %s - reconnecting",
                            symbol,
                            to_exc,
                        )
                        close_fn = getattr(exchange, "close", None)
                        if close_fn is None:
                            logger.warning("Exchange missing close method")
                        else:
                            try:
                                if asyncio.iscoroutinefunction(close_fn):
                                    await close_fn()
                                else:
                                    await asyncio.to_thread(close_fn)
                            except Exception as close_err:  # pragma: no cover - cleanup error
                                logger.error(
                                    "Exchange close failed: %s", close_err, exc_info=True
                                )
                        try:
                            exchange, _ = get_exchange(config)
                        except Exception as re_err:
                            reconnect_attempts += 1
                            logger.error(
                                "Reconnection attempt %d failed: %s",
                                reconnect_attempts,
                                re_err,
                                exc_info=True,
                            )
                            if reconnect_attempts >= max_reconnect:
                                logger.error(
                                    "WebSocket reconnection failed repeatedly; switching to REST"
                                )
                                use_ws = False
                            await asyncio.sleep(poll_interval)
                        continue
                    logger.warning(
                        "Ticker update timed out for %s - dropping", symbol
                    )
                    continue
                except asyncio.CancelledError:
                    raise
                except ccxt.RequestTimeout as ws_exc:
                    logger.warning(
                        "WebSocket ticker timeout for %s: %s - reconnecting",
                        symbol,
                        ws_exc,
                    )
                    close_fn = getattr(exchange, "close", None)
                    if close_fn is None:
                        logger.warning("Exchange missing close method")
                    else:
                        try:
                            if asyncio.iscoroutinefunction(close_fn):
                                await close_fn()
                            else:
                                await asyncio.to_thread(close_fn)
                        except Exception as close_err:  # pragma: no cover - cleanup error
                            logger.error(
                                "Exchange close failed: %s", close_err, exc_info=True
                            )
                    try:
                        exchange, _ = get_exchange(config)
                    except Exception as re_err:
                        reconnect_attempts += 1
                        logger.error(
                            "Reconnection attempt %d failed: %s",
                            reconnect_attempts,
                            re_err,
                            exc_info=True,
                        )
                        if reconnect_attempts >= max_reconnect:
                            logger.error("WebSocket reconnection failed repeatedly; switching to REST")
                            use_ws = False
                        await asyncio.sleep(poll_interval)
                    continue
                except Exception as ws_exc:  # pragma: no cover - websocket error
                    logger.error(
                        "WebSocket ticker failed for %s: %s - switching to REST",
                        symbol,
                        ws_exc,
                        exc_info=True,
                    )
                    use_ws = False
                    if ping_task:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass
                        WS_PING_TASKS.discard(ping_task)
                        ping_task = None
                    await asyncio.sleep(poll_interval)
                    continue
            else:
                if ping_task:
                    ping_task.cancel()
                    try:
                        await ping_task
                    except asyncio.CancelledError:
                        pass
                    WS_PING_TASKS.discard(ping_task)
                    ping_task = None
                if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ticker", None)):
                    ticker = await exchange.fetch_ticker(symbol)
                else:
                    ticker = await asyncio.to_thread(exchange.fetch_ticker, symbol)
                await asyncio.sleep(poll_interval)

            price = (
                ticker.get("last")
                or ticker.get("close")
                or ticker.get("bid")
                or ticker.get("ask")
            )
            if price is None:
                continue

            pos = state.positions.get(symbol)
            if pos is None:
                continue

            pos["last_price"] = price
            pos["pnl"] = (
                (price - pos["entry_price"])
                * pos["size"]
                * (1 if pos["side"] == "buy" else -1)
            )

            balance = await fetch_balance(exchange, paper_wallet, config)
            equity = balance
            if paper_wallet:
                equity = float(
                    paper_wallet.balance + paper_wallet.unrealized(symbol, price)
                )
            pos["equity"] = equity
        except asyncio.CancelledError:
            break
        except Exception as exc:  # pragma: no cover - network or runtime error
            logger.error("Watcher error for %s: %s", symbol, exc, exc_info=True)
            await asyncio.sleep(poll_interval)

    if ping_task:
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError:
            pass
        WS_PING_TASKS.discard(ping_task)


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
    """Fetch and store the current symbol batch."""
    symbols = await get_filtered_symbols(ctx.exchange, ctx.config)
    batch_size = ctx.config.get("symbol_batch_size", 10)
    ctx.current_batch = [s for s, _ in symbols][:batch_size]


async def update_caches(ctx: BotContext) -> None:
    """Update OHLCV and regime caches for the current batch."""
    batch = getattr(ctx, "current_batch", [])
    if not batch:
        return
    ctx.df_cache = await update_multi_tf_ohlcv_cache(
        ctx.exchange,
        ctx.df_cache,
        batch,
        ctx.config,
        limit=100,
        use_websocket=ctx.config.get("use_websocket", False),
        force_websocket_history=ctx.config.get("force_websocket_history", False),
        max_concurrent=ctx.config.get("max_concurrent_ohlcv"),
        notifier=None,
    )
    ctx.regime_cache = await update_regime_tf_cache(
        ctx.exchange,
        ctx.regime_cache,
        batch,
        ctx.config,
        limit=100,
        use_websocket=ctx.config.get("use_websocket", False),
        force_websocket_history=ctx.config.get("force_websocket_history", False),
        max_concurrent=ctx.config.get("max_concurrent_ohlcv"),
        notifier=None,
        df_map=ctx.df_cache,
    )


async def analyse_batch(ctx: BotContext) -> None:
    """Analyse the cached data for trading signals."""
    batch = getattr(ctx, "current_batch", [])
    if not batch:
        ctx.analysis_results = []
        return
    tasks = [
        analyze_symbol(
            sym,
            {tf: c.get(sym) for tf, c in ctx.df_cache.items()},
            "cex",
            ctx.config,
            None,
        )
        for sym in batch
    ]
    ctx.analysis_results = await asyncio.gather(*tasks)


async def execute_signals(ctx: BotContext) -> None:
    """Placeholder for executing trade signals."""
    ctx.executed = True


async def handle_exits(ctx: BotContext) -> None:
    """Placeholder for handling exit logic."""
    return


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
        await asyncio.sleep(interval)


async def _main_impl() -> TelegramNotifier:
    """Implementation for running the trading bot."""

    logger.info("Starting bot")
    config = load_config()
    metrics_path = (
        Path(config.get("metrics_csv")) if config.get("metrics_csv") else None
    )
    volume_ratio = 0.01 if config.get("testing_mode") else 1.0
    cooldown_configure(config.get("min_cooldown", 0))
    status_updates = config.get("telegram", {}).get("status_updates", True)
    market_loader_configure(
        config.get("ohlcv_timeout", 60),
        config.get("max_ohlcv_failures", 3),
        config.get("max_ws_limit", 50),
        status_updates,
    )
    secrets = dotenv_values(ENV_PATH)
    os.environ.update(secrets)

    user = load_or_create()

    trade_updates = config.get("telegram", {}).get("trade_updates", True)
    status_updates = config.get("telegram", {}).get("status_updates", True)
    balance_updates = config.get("telegram", {}).get("balance_updates", False)

    tg_cfg = {**config.get("telegram", {})}
    if user.get("telegram_token"):
        tg_cfg["token"] = user["telegram_token"]
    if user.get("telegram_chat_id"):
        tg_cfg["chat_id"] = user["telegram_chat_id"]
    if os.getenv("TELE_CHAT_ADMINS"):
        tg_cfg["chat_admins"] = os.getenv("TELE_CHAT_ADMINS")
    trade_updates = tg_cfg.get("trade_updates", True)
    status_updates = tg_cfg.get("status_updates", status_updates)
    balance_updates = tg_cfg.get("balance_updates", balance_updates)

    notifier = TelegramNotifier.from_config(tg_cfg)
    if status_updates:
        notifier.notify("🤖 CoinTrader2.0 started")

    if notifier.token and notifier.chat_id:
        if not send_test_message(notifier.token, notifier.chat_id, "Bot started"):
            logger.warning("Telegram test message failed; check your token and chat ID")

    # allow user-configured exchange to override YAML setting
    if user.get("exchange"):
        config["exchange"] = user["exchange"]

    exchange, ws_client = get_exchange(config)

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
    except Exception as exc:  # pragma: no cover - network
        logger.error("Exchange API setup failed: %s", exc)
        if status_updates:
            err = notifier.notify(f"API error: {exc}")
            if err:
                logger.error("Failed to notify user: %s", err)
        return
        previous_balance = float(init_bal)
    except Exception as exc:  # pragma: no cover - network
        logger.error("Exchange API setup failed: %s", exc)
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
    active_strategy = None
    stats_file = LOG_DIR / "strategy_stats.json"
    # File tracking individual trade performance used for analytics
    perf_file = LOG_DIR / "strategy_performance.json"
    scores_file = LOG_DIR / "strategy_scores.json"

    rotator = PortfolioRotator()
    last_optimize = 0.0
    last_weight_update = 0.0

    mode = user.get("mode", config.get("mode", "auto"))
    state = {"running": True, "mode": mode}
    df_cache: dict[str, dict[str, pd.DataFrame]] = {}
    regime_cache: dict[str, dict[str, pd.DataFrame]] = {}
    last_candle_ts: dict[str, int] = {}
    session_state = SessionState(last_balance=last_balance)

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

    await initial_scan(
        exchange,
        config,
        session_state,
        notifier if status_updates else None,
    )

    base_mode = mode
    loop_count = 0
    ctx = BotContext(positions, df_cache, regime_cache, config)
    ctx.exchange = exchange
    ctx.ws_client = ws_client
    runner = PhaseRunner([
        fetch_candidates,
        update_caches,
        analyse_batch,
        execute_signals,
        handle_exits,
    ])

    while True:
        loop_count += 1
        mode = state["mode"]
        maybe_update_mode(
            state, base_mode, config, notifier if status_updates else None
        )

        cycle_start = time.perf_counter()
        ctx.timing = await runner.run(ctx)
        execution_latency = 0.0

        total_pairs = 0
        signals_generated = 0
        trades_executed = 0
        rejected_volume = 0
        rejected_score = 0
        rejected_regime = 0
        volume_rejections = 0
        score_rejections = 0
        regime_rejections = 0

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
        for token in detect_non_trade_tokens(balances):
            amount = balances[token]
            logger.info("Converting %s %s to USDC", amount, token)
            await auto_convert_funds(
                user.get("wallet_address", ""),
                token,
                "USDC",
                amount,
                dry_run=config["execution_mode"] == "dry_run",
                slippage_bps=config.get("solana_slippage_bps", 50),
                notifier=notifier,
            )
            if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
                bal = await exchange.fetch_balance()
            else:
                bal = await asyncio.to_thread(exchange.fetch_balance)
            bal_val = (
                bal.get("USDT", {}).get("free", 0)
                if isinstance(bal.get("USDT"), dict)
                else bal.get("USDT", 0)
            )
            check_balance_change(float(bal_val), "funds converted")



        allowed_results: list[dict] = []
        df_current = None

        t0 = time.perf_counter()
        symbols = await get_filtered_symbols(exchange, config)
        symbol_time = time.perf_counter() - t0
        start_filter = time.perf_counter()
        global symbol_priority_queue
        if not symbol_priority_queue:
            symbol_priority_queue = build_priority_queue(symbols)
        ticker_fetch_time = time.perf_counter() - start_filter
        total_available = len(config.get("symbols") or [config.get("symbol")])
        symbol_filter_ratio = len(symbols) / total_available if total_available else 1.0
        global SYMBOL_EVAL_QUEUE
        if not SYMBOL_EVAL_QUEUE:
            SYMBOL_EVAL_QUEUE.extend(sym for sym, _ in symbols)
        batch_size = config.get("symbol_batch_size", 10)
        if len(symbol_priority_queue) < batch_size:
            symbol_priority_queue.extend(build_priority_queue(symbols))
        current_batch = [
            symbol_priority_queue.popleft()
            for _ in range(min(batch_size, len(symbol_priority_queue)))
        ]
        telemetry.inc("scan.symbols_considered", len(current_batch))

        t0 = time.perf_counter()
        start_ohlcv = time.perf_counter()
        tf_minutes = int(
            pd.Timedelta(config.get("timeframe", "1h")).total_seconds() // 60
        )
        limit = int(max(20, tf_minutes * 3))
        limit = int(config.get("cycle_lookback_limit", limit))

        session_state.df_cache = await update_multi_tf_ohlcv_cache(
            exchange,
            session_state.df_cache,
            current_batch,
            config,
            limit=limit,
            use_websocket=config.get("use_websocket", False),
            force_websocket_history=config.get("force_websocket_history", False),
            max_concurrent=config.get("max_concurrent_ohlcv"),
            notifier=notifier if status_updates else None,
        )

        session_state.regime_cache = await update_regime_tf_cache(
            exchange,
            session_state.regime_cache,
            current_batch,
            config,
            limit=limit,
            use_websocket=config.get("use_websocket", False),
            force_websocket_history=config.get("force_websocket_history", False),
            max_concurrent=config.get("max_concurrent_ohlcv"),
            notifier=notifier if status_updates else None,
            df_map=session_state.df_cache,
        )
        ohlcv_time = time.perf_counter() - t0
        ohlcv_fetch_latency = time.perf_counter() - start_ohlcv
        if notifier and ohlcv_fetch_latency > 0.5:
            notifier.notify(
                f"\u26a0\ufe0f OHLCV latency {ohlcv_fetch_latency*1000:.0f} ms"
            )

        tasks = []
        analyze_start = time.perf_counter()
        for sym in current_batch:
            logger.info("🔹 Symbol: %s", sym)
            total_pairs += 1
            df_map = {tf: c.get(sym) for tf, c in session_state.df_cache.items()}
            for tf, cache_tf in session_state.regime_cache.items():
                df_map[tf] = cache_tf.get(sym)
            df_sym = df_map.get(config["timeframe"])
            if df_sym is None or df_sym.empty:
                msg = (
                    f"OHLCV fetch failed for {sym} on {config['timeframe']} "
                    f"(limit {limit})"
                )
                logger.error(msg)
                if notifier and status_updates:
                    notifier.notify(msg)
                continue

            expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not isinstance(df_sym, pd.DataFrame):
                df_sym = pd.DataFrame(df_sym, columns=expected_cols)
            elif not set(expected_cols).issubset(df_sym.columns):
                df_sym = pd.DataFrame(df_sym.to_numpy(), columns=expected_cols)
            logger.info("Fetched %d candles for %s", len(df_sym), sym)
            df_map[config["timeframe"]] = df_sym
            if sym in session_state.positions:
                df_current = df_sym
            tasks.append(analyze_symbol(sym, df_map, mode, config, notifier))

        results = await asyncio.gather(*tasks)

        scalpers = [
            r["symbol"]
            for r in results
            if r.get("name") in {"micro_scalp", "bounce_scalper"}
        ]
        if scalpers:
            scalp_tf = config.get("scalp_timeframe", "1m")
            t_sc = time.perf_counter()
            session_state.df_cache[scalp_tf] = await update_ohlcv_cache(
                exchange,
                session_state.df_cache.get(scalp_tf, {}),
                scalpers,
                timeframe=scalp_tf,
                limit=100,
                use_websocket=config.get("use_websocket", False),
                force_websocket_history=config.get("force_websocket_history", False),
                config=config,
                max_concurrent=config.get("max_concurrent_ohlcv"),
            )
            ohlcv_time += time.perf_counter() - t_sc
            tasks = [
                analyze_symbol(
                    sym,
                    {
                        **{tf: c.get(sym) for tf, c in session_state.df_cache.items()},
                        **{tf: c.get(sym) for tf, c in session_state.regime_cache.items()},
                    },
                    mode,
                    config,
                    notifier,
                )
                for sym in scalpers
            ]
            scalper_results = await asyncio.gather(*tasks)
            mapping = {r["symbol"]: r for r in scalper_results}
            results = [mapping.get(r["symbol"], r) for r in results]
            for sym_open in session_state.positions:
                if sym_open in mapping:
                    current_dfs[sym_open] = session_state.df_cache.get(config["timeframe"], {}).get(
                        sym_open
                    )

        analyze_time = time.perf_counter() - analyze_start

        for res in results:
            sym = res["symbol"]
            df_sym = res["df"]
            regime_sym = res["regime"]
            log_regime(sym, regime_sym, res["future_return"])

            if regime_sym == "unknown":
                rejected_regime += 1
                continue

            env_sym = res["env"]
            name_sym = res["name"]
            score_sym = res["score"]
            direction_sym = res["direction"]

            logger.info("Regime %s -> Strategy %s", regime_sym, name_sym)
            logger.info(
                "Using strategy %s for %s in %s mode",
                name_sym,
                sym,
                env_sym,
            )

            if sym not in session_state.positions and in_cooldown(sym, name_sym):
                continue

            params_file = LOG_DIR / "optimized_params.json"
            if params_file.exists():
                params = json.loads(params_file.read_text())
                if name_sym in params:
                    risk_manager.config.stop_loss_pct = params[name_sym][
                        "stop_loss_pct"
                    ]
                    risk_manager.config.take_profit_pct = params[name_sym][
                        "take_profit_pct"
                    ]

            if direction_sym != "none":
                signals_generated += 1

            allowed, reason = risk_manager.allow_trade(df_sym, name_sym)
            mean_vol = df_sym["volume"].rolling(20).mean().iloc[-1]
            last_vol = df_sym["volume"].iloc[-1]
            logger.info(
                f"[TRADE EVAL] {sym} | Signal: {score_sym:.2f} | Volume: {last_vol:.4f}/{mean_vol:.2f} | Allowed: {allowed}"
            )
            if not allowed:
                logger.info("Trade not allowed for %s \u2013 %s", sym, reason)
                logger.info(
                    "Trade rejected for %s: %s, score=%.2f, regime=%s",
                    sym,
                    reason,
                    score_sym,
                    regime_sym,
                )
                if "Volume" in reason:
                    rejected_volume += 1
                    volume_rejections += 1
                else:
                    regime_rejections += 1
                continue

            base_min = config.get(
                "min_confidence_score", config.get("signal_threshold", 0.3)
            )
            min_score = res.get("min_confidence", base_min)
            if direction_sym != "none" and score_sym >= min_score:
                allowed_results.append(
                    {
                        "symbol": sym,
                        "df": df_sym,
                        "regime": regime_sym,
                        "env": env_sym,
                        "name": name_sym,
                        "direction": direction_sym,
                        "score": score_sym,
                        "atr": res.get("atr"),
                    }
                )

        allowed_results.sort(key=lambda x: x["score"], reverse=True)
        top_n = config.get("top_n_symbols", 3)
        allowed_results = allowed_results[:top_n]
        corr_matrix = compute_correlation_matrix(
            {r["symbol"]: r["df"] for r in allowed_results}
        )
        filtered_results: list[dict] = []
        for r in allowed_results:
            keep = True
            for kept in filtered_results:
                if not corr_matrix.empty:
                    corr = corr_matrix.at[r["symbol"], kept["symbol"]]
                    if abs(corr) > 0.95:
                        keep = False
                        break
            if keep:
                filtered_results.append(r)

        best = filtered_results[0] if filtered_results else None

        current_dfs: dict[str, pd.DataFrame] = {}
        current_prices: dict[str, float] = {}

        open_syms = list(session_state.positions.keys())
        if open_syms:
            tf_cache = df_cache.get(config["timeframe"], {})
            update_syms: list[str] = []
            for s in open_syms:
                ts = None
                df_prev = tf_cache.get(s)
                if df_prev is not None and not df_prev.empty:
                    ts = int(df_prev["timestamp"].iloc[-1])
                if ts is None or last_candle_ts.get(s) != ts:
                    update_syms.append(s)
            if update_syms:
                tf_cache = await update_ohlcv_cache(
                    exchange,
                    tf_cache,
                    update_syms,
                    timeframe=config["timeframe"],
                    limit=100,
                    use_websocket=config.get("use_websocket", False),
                    force_websocket_history=config.get("force_websocket_history", False),
                    config=config,
                    max_concurrent=config.get("max_concurrent_ohlcv"),
                )
                for s in update_syms:
                    df_new = tf_cache.get(s)
                    if df_new is not None and not df_new.empty:
                        last_candle_ts[s] = int(df_new["timestamp"].iloc[-1])
                df_cache[config["timeframe"]] = tf_cache
            session_state.df_cache[config["timeframe"]] = await update_ohlcv_cache(
                exchange,
                session_state.df_cache.get(config["timeframe"], {}),
                open_syms,
                timeframe=config["timeframe"],
                limit=100,
                use_websocket=config.get("use_websocket", False),
                force_websocket_history=config.get("force_websocket_history", False),
                config=config,
                max_concurrent=config.get("max_concurrent_ohlcv"),
            )

        for sym in open_syms:
            df_current = session_state.df_cache.get(config["timeframe"], {}).get(sym)
            if df_current is None:
                # Fallback to direct fetch if cache is missing
                try:
                    if config.get("use_websocket", False) and hasattr(
                        exchange, "watch_ohlcv"
                    ):
                        data = await exchange.watch_ohlcv(
                            sym,
                            timeframe=config["timeframe"],
                            limit=100,
                        )
                    else:
                        if asyncio.iscoroutinefunction(
                            getattr(exchange, "fetch_ohlcv", None)
                        ):
                            data = await exchange.fetch_ohlcv(
                                sym,
                                timeframe=config["timeframe"],
                                limit=100,
                            )
                        else:
                            data = await asyncio.to_thread(
                                exchange.fetch_ohlcv,
                                sym,
                                timeframe=config["timeframe"],
                                limit=100,
                            )
                except Exception as exc:  # pragma: no cover - network
                    logger.error(
                        "OHLCV fetch failed for %s on %s (limit %d): %s",
                        sym,
                        config["timeframe"],
                        100,
                        exc,
                        exc_info=True,
                    )
                    continue

                if data and len(data[0]) > 6:
                    data = [[c[0], c[1], c[2], c[3], c[4], c[6]] for c in data]
                df_current = pd.DataFrame(
                    data,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df_cache.setdefault(config["timeframe"], {})[sym] = df_current
            if df_current is not None and not df_current.empty:
                last_candle_ts[sym] = int(df_current["timestamp"].iloc[-1])
                session_state.df_cache.setdefault(config["timeframe"], {})[sym] = df_current

            current_dfs[sym] = df_current
            current_prices[sym] = df_current["close"].iloc[-1]

        df_current = current_dfs.get(best["symbol"] if best else "")
        current_price = None
        if best:
            if df_current is None:
                df_current = best["df"]
            current_price = df_current["close"].iloc[-1]

        if best:
            score = best["score"]
            direction = best["direction"]
            trade_side = direction_to_side(direction)
            env = best["env"]
            regime = best["regime"]
            name = best["name"]
        else:
            score = -1
            direction = "none"
            trade_side = None

        for sym, pos in list(session_state.positions.items()):
            cur_price = current_prices.get(sym)
            df_cur = current_dfs.get(sym)
            if cur_price is None or df_cur is None:
                continue
            pnl_pct = ((cur_price - pos["entry_price"]) / pos["entry_price"]) * (
                1 if pos["side"] == "buy" else -1
            )
            if pnl_pct >= config["exit_strategy"]["min_gain_to_trail"]:
                if cur_price > pos.get("highest_price", pos["entry_price"]):
                    pos["highest_price"] = cur_price
                pos["trailing_stop"] = calculate_trailing_stop(
                    pd.Series([pos.get("highest_price", cur_price)]),
                    config["exit_strategy"]["trailing_stop_pct"],
                )
            risk_manager.stop_order = risk_manager.get_stop_order(sym)
            exit_signal, new_stop = should_exit(
                df_cur,
                cur_price,
                pos.get("trailing_stop", 0.0),
                config,
                risk_manager,
            )
            pos["trailing_stop"] = new_stop
            equity = paper_wallet.balance if paper_wallet else latest_balance
            if paper_wallet:
                unreal = paper_wallet.unrealized(sym, cur_price)
                equity += unreal
            pos["equity"] = float(equity)
            session_state.last_balance = notify_balance_change(
                notifier,
                session_state.last_balance,
                float(equity),
                balance_updates,
            )
            if exit_signal:
                pct = get_partial_exit_percent(pnl_pct * 100)
                sell_amount = (
                    pos["size"] * (pct / 100)
                    if config["exit_strategy"]["scale_out"] and pct > 0
                    else pos["size"]
                )
                _start_exec = time.perf_counter()
                await cex_trade_async(
                    exchange,
                    ws_client,
                    sym,
                    opposite_side(pos["side"]),
                    sell_amount,
                    notifier,
                    dry_run=config["execution_mode"] == "dry_run",
                    use_websocket=config.get("use_websocket", False),
                    config=config,
                )
                lat = time.perf_counter() - _start_exec
                execution_latency = max(execution_latency, lat)
                if notifier and lat > 0.5:
                    notifier.notify(f"\u26a0\ufe0f Execution latency {lat*1000:.0f} ms")
                if paper_wallet:
                    paper_wallet.close(sym, sell_amount, cur_price)
                pos["pnl"] = pos.get("pnl", 0.0) + (
                    (cur_price - pos["entry_price"])
                    * sell_amount
                    * (1 if pos["side"] == "buy" else -1)
                )
                if sell_amount >= pos["size"]:
                    risk_manager.cancel_stop_order(exchange, sym)
                    risk_manager.deallocate_capital(
                        pos["strategy"], sell_amount * pos["entry_price"]
                    )
                if sell_amount >= position_size:
                    risk_manager.cancel_stop_order(
                        exchange, open_symbol or config.get("symbol", "")
                    )
                    risk_manager.deallocate_capital(
                        current_strategy, sell_amount * entry_price
                    )
                    log_performance(
                        {
                            "symbol": sym,
                            "regime": pos.get("regime"),
                            "strategy": pos.get("strategy"),
                            "pnl": pos["pnl"],
                            "entry_time": pos.get("entry_time"),
                            "exit_time": datetime.utcnow().isoformat(),
                        }
                    )
                    log_pnl(
                        pos.get("strategy", ""),
                        sym,
                        pos["entry_price"],
                        cur_price,
                        pos["pnl"],
                        pos.get("confidence", 0.0),
                        pos["side"],
                    )
                    log_regime_pnl(
                        pos.get("regime", "unknown"),
                        pos.get("strategy", ""),
                        pos["pnl"],
                    )
                    latest_balance = await fetch_balance(exchange, paper_wallet, config)
                    log_position(
                        sym,
                        pos["side"],
                        sell_amount,
                        pos["entry_price"],
                        cur_price,
                        float(paper_wallet.balance if paper_wallet else latest_balance),
                    )
                    mark_cooldown(sym, active_strategy or pos.get("strategy", ""))
                    task = position_tasks.pop(sym, None)
                    if task:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    positions.pop(sym, None)
                    last_candle_ts.pop(sym, None)
                    session_state.positions.pop(sym, None)
                    latest_balance = await fetch_and_log_balance(
                        exchange, paper_wallet, config
                    )
                    equity = paper_wallet.balance if paper_wallet else latest_balance
                    log_position(
                        sym,
                        pos["side"],
                        pos["size"],
                        pos["entry_price"],
                        cur_price,
                        float(equity),
                        pnl=pos["pnl"],
                    )
                else:
                    pos["size"] -= sell_amount
                    risk_manager.deallocate_capital(
                        pos["strategy"], sell_amount * pos["entry_price"]
                    )
                    risk_manager.update_stop_order(pos["size"], symbol=sym)
                    latest_balance = await fetch_balance(exchange, paper_wallet, config)
                    position_size -= sell_amount
                    risk_manager.deallocate_capital(
                        current_strategy, sell_amount * entry_price
                    )
                    risk_manager.update_stop_order(
                        open_symbol or config.get("symbol", ""), position_size
                    )
                    latest_balance = await fetch_balance(exchange, paper_wallet, config)

        if not position_guard.can_open(session_state.positions):
            continue

        if not filtered_results:
            continue

        for candidate in filtered_results:
            if candidate["symbol"] in session_state.positions:
                logger.info(
                    "Existing position on %s still open – skipping new trade",
                    candidate["symbol"],
                )
                continue
            if not position_guard.can_open(session_state.positions):
                break
            score = candidate["score"]
            direction = candidate["direction"]
            trade_side = direction_to_side(direction)
            env = candidate["env"]
            regime = candidate["regime"]
            name = candidate["name"]
            current_price = candidate["df"]["close"].iloc[-1]
            risk_manager.config.symbol = candidate["symbol"]
            df_for_size = candidate["df"]

            if score < config["signal_threshold"]:
                rejected_score += 1
                score_rejections += 1
            logger.info(
                "Cycle Summary: %s pairs evaluated, %s signals, %s trades executed, %s rejected volume, %s rejected score, %s rejected regime.",
                total_pairs,
                signals_generated,
                trades_executed,
                volume_rejections,
                score_rejections,
                regime_rejections,
            )
            if config.get("metrics_enabled") and config.get("metrics_backend") == "csv":
                metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "ticker_fetch_time": ticker_fetch_time,
                    "symbol_filter_ratio": symbol_filter_ratio,
                    "ohlcv_fetch_latency": ohlcv_fetch_latency,
                    "execution_latency": execution_latency,
                    "unknown_regimes": rejected_regime,
                }
                log_metrics_to_csv(
                    metrics,
                    config.get("metrics_output_file", str(LOG_DIR / "metrics.csv")),
                )
            logger.info("Sleeping for %s minutes", config["loop_interval_minutes"])
            await asyncio.sleep(config["loop_interval_minutes"] * 60)

            poll_mod = config.get("balance_poll_mod", 1)
            balance = last_balance if last_balance is not None else 0.0
            if trades_executed > 0 or loop_count % poll_mod == 0:
                if config["execution_mode"] != "dry_run":
                    bal = await (
                        exchange.fetch_balance()
                        if asyncio.iscoroutinefunction(
                            getattr(exchange, "fetch_balance", None)
                        )
                        else asyncio.to_thread(exchange.fetch_balance)
                    )
                    balance = bal["USDT"]["free"]
                else:
                    balance = paper_wallet.balance if paper_wallet else 0.0
                check_balance_change(float(balance), "external change")

            size = risk_manager.position_size(
                score,
                balance,
                df_for_size,
                atr=candidate.get("atr"),
                price=current_price,
            )
            order_amount = size / current_price if current_price > 0 else 0.0

            if not risk_manager.can_allocate(name, size, balance):
                logger.info("Capital cap reached for %s, skipping", name)
                logger.info(
                    "Loop Summary: %s evaluated | %s trades | %s volume fails | %s score fails | %s unknown regime",
                    total_pairs,
                    trades_executed,
                    rejected_volume,
                    rejected_score,
                    rejected_regime,
                )
                logger.info(
                    "Cycle Summary: %s pairs evaluated, %s signals, %s trades executed, %s rejected volume, %s rejected score, %s rejected regime.",
                    total_pairs,
                    signals_generated,
                    trades_executed,
                    volume_rejections,
                    score_rejections,
                    regime_rejections,
                )
                if (
                    config.get("metrics_enabled")
                    and config.get("metrics_backend") == "csv"
                ):
                    metrics = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "ticker_fetch_time": ticker_fetch_time,
                        "symbol_filter_ratio": symbol_filter_ratio,
                        "ohlcv_fetch_latency": ohlcv_fetch_latency,
                        "execution_latency": execution_latency,
                        "unknown_regimes": rejected_regime,
                    }
                    log_metrics_to_csv(
                        metrics,
                        config.get(
                            "metrics_output_file", str(LOG_DIR / "metrics.csv")
                        ),
                    )
                logger.info("Sleeping for %s minutes", config["loop_interval_minutes"])
                await asyncio.sleep(config["loop_interval_minutes"] * 60)
                continue

            _start_exec = time.perf_counter()
            order = await cex_trade_async(
                exchange,
                ws_client,
                candidate["symbol"],
                trade_side,
                order_amount,
                notifier,
                dry_run=config["execution_mode"] == "dry_run",
                use_websocket=config.get("use_websocket", False),
                config=config,
            )
            lat = time.perf_counter() - _start_exec
            execution_latency = max(execution_latency, lat)
            if notifier and lat > 0.5:
                notifier.notify(f"\u26a0\ufe0f Execution latency {lat*1000:.0f} ms")
            atr_val = candidate.get("atr")
            if atr_val:
                stop_price = (
                    current_price - atr_val * risk_manager.config.stop_loss_atr_mult
                    if trade_side == "buy"
                    else current_price
                    + atr_val * risk_manager.config.stop_loss_atr_mult
                )
                take_profit_price = (
                    current_price + atr_val * risk_manager.config.take_profit_atr_mult
                    if trade_side == "buy"
                    else current_price
                    - atr_val * risk_manager.config.take_profit_atr_mult
                )
            else:
                stop_price = current_price * (
                    1 - risk_manager.config.stop_loss_pct
                    if trade_side == "buy"
                    else 1 + risk_manager.config.stop_loss_pct
                )
                take_profit_price = current_price * (
                    1 + risk_manager.config.take_profit_pct
                    if trade_side == "buy"
                    else 1 - risk_manager.config.take_profit_pct
                )
            stop_order = place_stop_order(
                exchange,
                candidate["symbol"],
                "sell" if trade_side == "buy" else "buy",
                order_amount,
                stop_price,
                notifier=notifier,
                dry_run=config["execution_mode"] == "dry_run",
            )
            risk_manager.register_stop_order(
                stop_order,
                strategy=strategy_name(regime, env),
                symbol=candidate["symbol"],
                entry_price=current_price,
                confidence=score,
                direction=trade_side,
                take_profit=take_profit_price,
            )
            risk_manager.allocate_capital(name, size)
            if config["execution_mode"] == "dry_run" and paper_wallet:
                paper_wallet.open(
                    candidate["symbol"], trade_side, order_amount, current_price
                )
                latest_balance = paper_wallet.balance
            else:
                if asyncio.iscoroutinefunction(
                    getattr(exchange, "fetch_balance", None)
                ):
                    bal = await exchange.fetch_balance()
                else:
                    bal = await asyncio.to_thread(exchange.fetch_balance)
                latest_balance = (
                    bal["USDT"]["free"]
                    if isinstance(bal["USDT"], dict)
                    else bal["USDT"]
                )
            check_balance_change(float(latest_balance), "trade executed")
            log_balance(float(latest_balance))
            session_state.last_balance = notify_balance_change(
                notifier,
                session_state.last_balance,
                float(latest_balance),
                balance_updates,
            )
            log_position(
                candidate["symbol"],
                trade_side,
                order_amount,
                current_price,
                current_price,
                float(latest_balance),
            )
            if strategy_name(regime, env).startswith("grid"):
                grid_state.record_fill(candidate["symbol"])
            session_state.positions[candidate["symbol"]] = {
                "side": trade_side,
                "entry_price": current_price,
                "entry_time": datetime.utcnow().isoformat(),
                "regime": regime,
                "strategy": strategy_name(regime, env),
                "confidence": score,
                "pnl": 0.0,
                "size": order_amount,
                "trailing_stop": 0.0,
                "highest_price": current_price,
            }
            position_tasks[candidate["symbol"]] = asyncio.create_task(
                _watch_position(
                    candidate["symbol"],
                    exchange,
                    session_state,
                    paper_wallet,
                    config,
                )
            )
            active_strategy = name
            if notifier and trade_updates:
                report_entry(
                    notifier,
                    candidate["symbol"],
                    strategy_name(regime, env),
                    score,
                    direction,
                )
            logger.info("Trade opened at %.4f", current_price)
            trades_executed += 1
            if not position_guard.can_open(session_state.positions):
                break

        write_scores(scores_file, perf_file)
        write_stats(stats_file, perf_file)

        logger.info(
            "Loop Summary: %s evaluated | %s trades | %s volume fails | %s score fails | %s unknown regime",
            total_pairs,
            trades_executed,
            rejected_volume,
            rejected_score,
            rejected_regime,
        )
        logger.info(
            "Cycle Summary: %s pairs evaluated, %s signals, %s trades executed, %s rejected volume, %s rejected score, %s rejected regime.",
            total_pairs,
            signals_generated,
            trades_executed,
            volume_rejections,
            score_rejections,
            regime_rejections,
        )
        total_time = time.perf_counter() - cycle_start
        timing = getattr(ctx, "timing", {})
        _emit_timing(
            timing.get("fetch_candidates", symbol_time),
            timing.get("update_caches", ohlcv_time),
            timing.get("analyse_batch", analyze_time),
            total_time,
            metrics_path,
            ohlcv_fetch_latency,
            execution_latency,
        )
        if config.get("metrics_enabled") and config.get("metrics_backend") == "csv":
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "ticker_fetch_time": ticker_fetch_time,
                "symbol_filter_ratio": symbol_filter_ratio,
                "ohlcv_fetch_latency": ohlcv_fetch_latency,
                "execution_latency": execution_latency,
                "unknown_regimes": rejected_regime,
            }
            log_metrics_to_csv(
                metrics,
                config.get("metrics_output_file", str(LOG_DIR / "metrics.csv")),
            )
            telemetry.export_csv(
                config.get("metrics_output_file", str(LOG_DIR / "telemetry.csv"))
            )
        summary = f"Cycle complete: {total_pairs} symbols, {trades_executed} trades"
        if notifier and status_updates:
            notifier.notify(summary)
        logger.info("Sleeping for %s minutes", config["loop_interval_minutes"])
        await asyncio.sleep(config["loop_interval_minutes"] * 60)

    monitor_task.cancel()
    control_task.cancel()
    rotation_task.cancel()
    for task in list(position_tasks.values()):
        task.cancel()
    for task in list(position_tasks.values()):
        try:
            await task
        except asyncio.CancelledError:
            pass
    position_tasks.clear()
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
            await exchange.close()
        else:
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
