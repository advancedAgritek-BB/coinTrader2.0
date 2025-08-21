from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
import dataclasses
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
from dotenv import dotenv_values, load_dotenv


try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ccxt = types.SimpleNamespace()

import pandas as pd
import numpy as np
import yaml
from types import SimpleNamespace
from pydantic import ValidationError
from crypto_bot.utils.logger import pipeline_logger, LOG_DIR, setup_logger
from crypto_bot.utils.logging_config import setup_logging

lastlog = setup_logging(LOG_DIR / "bot.log")

from crypto_bot.universe import build_tradable_set
from crypto_bot.strategy.evaluator import StreamEvaluator, set_stream_evaluator
from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
from crypto_bot.ml.selfcheck import log_ml_status_once
from crypto_bot.utils import ml_utils

# Expose ML availability flag for tests and external modules
ML_AVAILABLE = ml_utils.ML_AVAILABLE
from crypto_bot.ml.model_loader import load_regime_model, _norm_symbol

# Internal project modules are imported lazily in `_import_internal_modules()`

try:  # pragma: no cover - optional strategies module
    from crypto_bot import strategies
    from crypto_bot.strategies import set_ohlcv_provider
    from crypto_bot.strategies.loader import load_strategies
except Exception:  # pragma: no cover - fallback if strategies module missing
    strategies = types.SimpleNamespace(
        initialize=lambda *_a, **_k: asyncio.sleep(0),
        score=lambda *_a, **_k: {},
    )

    def set_ohlcv_provider(*_a, **_k):
        """Fallback no-op when strategies module is unavailable."""
        return None

    def load_strategies(*_a, **_k):
        return []

shutdown_event = asyncio.Event()


def format_top(scores, n: int = 25) -> str:
    """Return formatted top-N entries from a score mapping."""
    try:
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return "\n".join(f"{sym}: {val}" for sym, val in items)
    except Exception:
        return str(scores)


logger = logging.getLogger("bot")
pythonlogger = logging.getLogger("python")

# Dedicated logger for symbol scoring
score_logger = setup_logger(
    "symbol_filter", LOG_DIR / "symbol_filter.log", to_console=False
)

# Module-level placeholders populated once internal modules are loaded in ``main``


def build_priority_queue(scores):
    return deque(
        sym for sym, _ in sorted(scores, key=lambda x: x[1], reverse=True)
    )  # type: ignore


get_solana_new_tokens = None  # type: ignore
get_filtered_symbols = None  # type: ignore


async def fetch_from_helius(*_a, **_k):
    return {}


from crypto_bot.utils import symbol_utils

fix_symbol = symbol_utils.fix_symbol
calc_atr = None  # type: ignore


def timeframe_seconds(*_a, **_k) -> int:
    return 0  # type: ignore


maybe_refresh_model = None  # type: ignore
registry = None  # type: ignore
fetch_geckoterminal_ohlcv = None  # type: ignore
fetch_solana_prices = None  # type: ignore
cross_chain_trade = None  # type: ignore
sniper_solana = None  # type: ignore
sniper_trade = None  # type: ignore
load_token_mints = None  # type: ignore
set_token_mints = None  # type: ignore
TelegramBotUI = None  # type: ignore
start_runner = None  # type: ignore
sniper_run = None  # type: ignore
load_or_create = None  # type: ignore
send_test_message = None  # type: ignore
log_balance = None  # type: ignore
get_exchange = None  # type: ignore
load_kraken_symbols = None  # type: ignore


def cooldown_configure(*_a, **_k) -> None:  # pragma: no cover - placeholder
    pass


update_ohlcv_cache = None  # type: ignore
update_multi_tf_ohlcv_cache = None  # type: ignore
update_regime_tf_cache = None  # type: ignore


def collect_timeframes(config: dict) -> list[str]:
    """Return sorted unique timeframes required by strategies."""
    tfs: set[str] = set(config.get("timeframes", []))
    router_cfg = config.get("strategy_router", {})
    for key, val in router_cfg.items():
        if key.endswith("_timeframe") and isinstance(val, str):
            tfs.add(val)
    return sorted(tfs, key=lambda t: timeframe_seconds(None, t))


def market_loader_configure(*_a, **_k) -> None:  # pragma: no cover - placeholder
    pass


fetch_order_book_async = None  # type: ignore
WS_OHLCV_TIMEOUT = None  # type: ignore
ScannerConfig = None  # type: ignore
SolanaScannerConfig = None  # type: ignore
PythConfig = None  # type: ignore
stream_evaluator: StreamEvaluator | None = None


def build_risk_config(config: dict, volume_ratio: float) -> RiskConfig:
    """Construct a :class:`RiskConfig` from config sections."""
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
    fields = {f.name for f in dataclasses.fields(RiskConfig)}
    risk_params = {k: v for k, v in risk_params.items() if k in fields}
    return RiskConfig(**risk_params)


class BotContext:
    def __init__(self, _a=None, _b=None, _c=None, config=None):
        self.config = config or {}


class OpenPositionGuard:
    def __init__(self, max_open_trades: int):
        self.max_open_trades = max_open_trades


@dataclass
class MainResult:
    """Result returned from ``_main_impl`` summarizing shutdown state."""

    notifier: "TelegramNotifier"
    reason: str


async def evaluation_loop(
    run_cycle: Callable[[], Awaitable[None]],
    ctx: Any,
    config: dict,
    stop_reason_ref: list[str],
) -> None:
    """Continuously run evaluation cycles with timeout and logging."""
    loop_interval = config.get("loop_interval_minutes", 1)
    eval_timeout = config.get("evaluation_timeout", loop_interval * 60)
    while not shutdown_event.is_set():
        logger.info("Starting evaluation cycle")
        try:
            await asyncio.wait_for(run_cycle(), eval_timeout)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            logger.error(
                "Evaluation cycle timed out after %.0f seconds", eval_timeout
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception("Evaluation cycle failed: %s", exc)
            if stop_reason_ref[0] == "completed":
                stop_reason_ref[0] = f"evaluation cycle failed: {exc}"
        logger.info(
            "Active universe: %d symbols, current batch: %d symbols",
            len(getattr(ctx, "active_universe", []) or []),
            len(getattr(ctx, "current_batch", []) or []),
        )
        await asyncio.sleep(loop_interval * 60)


PhaseRunner = None  # type: ignore
calculate_trailing_stop = None  # type: ignore
should_exit = None  # type: ignore


@contextlib.asynccontextmanager
async def symbol_cache_guard():
    """Fallback cache guard before internal modules are loaded."""
    yield


# _fix_symbol is defined below for backward compatibility

REQUIRED_ENV_VARS = {
    "HELIUS_KEY",
    "SOLANA_PRIVATE_KEY",
    "TELEGRAM_TOKEN",
    "TELEGRAM_CHAT_ID",
}

CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CONFIG_DIR / "config.yaml"
ENV_PATH = CONFIG_DIR / ".env"
USER_CONFIG_PATH = CONFIG_DIR / "user_config.yaml"


def _norm(v: Any):
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in {"", "none", "null"}:
        return None
    return v


def _load_env(path: Path = ENV_PATH) -> dict[str, str | None]:
    """Load environment variables from ``path`` into ``os.environ``."""
    env = {k: _norm(v) for k, v in dotenv_values(path).items()}
    for key, value in env.items():
        if key not in os.environ and value is not None:
            os.environ[key] = str(value)
    return env


def _read_yaml(path: Path) -> dict:
    """Return parsed YAML from ``path`` or an empty dict if missing."""
    try:
        with path.open() as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _has_env(env: dict[str, str | None], key: str) -> bool:
    """Return ``True`` if ``key`` exists in ``env`` or ``os.environ``."""
    return bool(env.get(key) or os.getenv(key))


def _needs_wallet_setup(
    env: dict[str, str | None], cfg_path: Path = USER_CONFIG_PATH
) -> bool:
    """Determine whether wallet configuration is missing credentials."""
    cfg = _read_yaml(cfg_path) if cfg_path.exists() else {}
    has_api = any(
        _has_env(env, key) or cfg.get(key.lower())
        for key in ("API_KEY", "COINBASE_API_KEY", "KRAKEN_API_KEY")
    )
    return not (cfg_path.exists() and has_api)


def _run_wallet_manager() -> None:
    """Execute the wallet manager or guide the user in non-interactive mode."""
    if not sys.stdin.isatty():
        print(
            "Interactive setup required but no TTY is attached.\n"
            "Run `python -m crypto_bot.wallet_manager` once to create credentials, "
            "or set them in the repository .env file.",
            file=sys.stderr,
        )
        sys.exit(2)
    result = subprocess.run([sys.executable, "-m", "crypto_bot.wallet_manager"])
    if result.returncode not in (0, None):
        sys.exit(result.returncode)


def _ensure_user_setup() -> None:
    """Ensure required credentials exist or launch the wallet setup wizard."""
    _load_env()
    if USER_CONFIG_PATH.exists() and all(os.getenv(var) for var in REQUIRED_ENV_VARS):
        return
    _run_wallet_manager()
    _load_env()
    env = _load_env()
    if _needs_wallet_setup(env, USER_CONFIG_PATH) or not all(
        os.getenv(var) for var in REQUIRED_ENV_VARS
    ):
        _run_wallet_manager()
        _load_env()
    if _needs_wallet_setup(env):
        _run_wallet_manager()


def _fix_symbol(symbol: str) -> str:
    """Backward compatible wrapper around :func:`fix_symbol`."""
    if fix_symbol is None:  # pragma: no cover - should be loaded in main
        raise RuntimeError("fix_symbol not loaded")
    return fix_symbol(symbol)


# In-memory cache of configuration and file mtimes
_CONFIG_CACHE: dict[str, object] = {}
_CONFIG_MTIMES: dict[Path, tuple[float, int]] = {}
# Track ML-related settings to avoid re-loading the model unnecessarily
_LAST_ML_CFG: dict[str, object] | None = None


class MLUnavailableError(RuntimeError):
    """Raised when required ML components are missing or fail to load."""

    def __init__(self, message: str, cfg: dict | None = None) -> None:
        super().__init__(message)
        self.cfg = cfg


# Track WebSocket ping tasks
WS_PING_TASKS: set[asyncio.Task] = set()
# Track async sniper trade tasks
SNIPER_TASKS: set[asyncio.Task] = set()
# Track newly scanned Solana tokens pending evaluation
NEW_SOLANA_TOKENS: set[str] = set()
# Track async cross-chain arb tasks
CROSS_ARB_TASKS: set[asyncio.Task] = set()
# Track all spawned background tasks for coordinated shutdown
BACKGROUND_TASKS: list[asyncio.Task] = []
# Track pending OHLCV tasks during startup
pending_tasks: list[asyncio.Task] = []
# Track outstanding bootstrap tasks between cycles
BOOTSTRAP_TASKS: set[asyncio.Task] = set()

# Track background task failures to surface repeated issues
TASK_FAILURE_COUNTS: Counter[str] = Counter()
TASK_FAILURE_NOTIFIER: "TelegramNotifier" | None = None
TASK_FAILURE_NOTIFY_THRESHOLD = 3


def register_task(task: asyncio.Task | None) -> asyncio.Task | None:
    """Add a task to the background task registry."""
    if not task:
        return None

    BACKGROUND_TASKS.append(task)

    def _handle_task_completion(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is None:
            return
        name = t.get_name()
        logger.error("Background task %s raised an exception", name, exc_info=exc)
        TASK_FAILURE_COUNTS[name] += 1
        if (
            TASK_FAILURE_NOTIFIER
            and TASK_FAILURE_COUNTS[name] >= TASK_FAILURE_NOTIFY_THRESHOLD
        ):
            try:
                TASK_FAILURE_NOTIFIER.notify(
                    f"Task {name} failed {TASK_FAILURE_COUNTS[name]} times: {exc}"
                )
            except Exception:
                logger.exception("Failed to send failure notification for %s", name)

    task.add_done_callback(_handle_task_completion)
    return task


def _schedule_bootstrap(coro: Awaitable[Any]) -> None:
    """Schedule ``coro`` as a background bootstrap task."""
    task = register_task(asyncio.create_task(coro))
    if task:
        BOOTSTRAP_TASKS.add(task)


async def _run_bootstrap(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    run_timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    """Run ``func`` with ``asyncio.wait_for`` and queue background work on timeout.

    If the call times out or returns a tuple ``(result, remaining)`` with
    ``remaining`` truthy, a background task is scheduled to continue processing
    using the same arguments. The primary result is returned when available.
    """
    if run_timeout is not None and run_timeout <= 0:
        return await func(*args, **kwargs)
    try:
        result = await asyncio.wait_for(func(*args, **kwargs), run_timeout)
    except asyncio.TimeoutError:
        _schedule_bootstrap(func(*args, **kwargs))
        return None
    else:
        if isinstance(result, tuple) and len(result) == 2:
            primary, remaining = result
            if remaining:
                _schedule_bootstrap(func(*args, **kwargs))
            return primary
        return result


def _prune_bootstrap_tasks() -> None:
    """Remove completed bootstrap tasks from the tracker."""
    for task in list(BOOTSTRAP_TASKS):
        if task.done():
            BOOTSTRAP_TASKS.discard(task)


# Queue of symbols awaiting evaluation across loops
symbol_priority_queue: deque[str] = deque()

# Symbols that produced no OHLCV data and should be skipped until refreshed
no_data_symbols: set[str] = set()

# Cache of recently queued Solana tokens to avoid duplicates
SOLANA_CACHE_SIZE = 50
recent_solana_tokens: deque[str] = deque()
recent_solana_set: set[str] = set()

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
        atr_series = calc_atr(df)
        atr_val = float(atr_series.iloc[-1]) if len(atr_series) else np.nan
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        atr_values.append(atr_val)
    return sum(atr_values) / len(atr_values) if atr_values else 0.0


def format_monitor_line(
    ctx: Any,
    session_state: "SessionState",
    balance: float,
    positions: dict[str, Any],
    last_log: str,
) -> str:
    """Return a formatted status line with OHLCV progress and IOPS."""
    tickers = ",".join(sorted(positions.keys())) or "-"
    active = getattr(ctx, "active_universe", [])
    total = len(active)
    parts: list[str] = []
    for tf in ctx.config.get("timeframes", []):
        tf_cache = session_state.df_cache.get(tf, {})
        count = sum(1 for s in active if s in tf_cache)
        parts.append(f"{tf}: {count}/{total}")
    ohlcv_summary = " | ".join(parts) if parts else "-"
    from crypto_bot.utils import market_loader

    iops = market_loader.get_iops()
    return (
        f"[Monitor] balance=${balance:,.2f} open={len(positions)} ({tickers}) "
        f"last='{last_log}' OHLCV {ohlcv_summary} IOPS {iops:.1f}/s"
    )


def enqueue_solana_tokens(tokens: list[str]) -> None:
    """Add Solana ``tokens`` to the priority queue skipping recently queued ones."""

    for sym in reversed(tokens):
        if sym in recent_solana_set:
            continue
        symbol_priority_queue.appendleft(sym)
        recent_solana_tokens.append(sym)
        recent_solana_set.add(sym)
        if len(recent_solana_tokens) > SOLANA_CACHE_SIZE:
            old = recent_solana_tokens.popleft()
            recent_solana_set.discard(old)


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
    label, info = await classify_regime_cached(
        sym, base_tf, df, higher_df, notifier=ctx.notifier
    )
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


async def fetch_balance(exchange, wallet, config):
    """Return the latest wallet balance without logging."""
    if config["execution_mode"] != "dry_run":
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
            bal = await exchange.fetch_balance()
        else:
            bal = await asyncio.to_thread(exchange.fetch_balance)
        return bal["USDT"]["free"] if isinstance(bal["USDT"], dict) else bal["USDT"]
    if wallet:
        return getattr(wallet, "total_balance", getattr(wallet, "balance", 0.0))
    return 0.0


async def fetch_and_log_balance(exchange, wallet, config):
    """Return the latest wallet balance and log it."""
    latest_balance = await fetch_balance(exchange, wallet, config)
    log_balance(float(latest_balance))
    return latest_balance


async def refresh_balance(ctx: BotContext) -> float:
    """Update ``ctx.balance`` from the exchange or paper wallet."""
    latest = await fetch_and_log_balance(
        ctx.exchange,
        ctx.wallet,
        ctx.config,
    )
    ctx.balance = notify_balance_change(
        ctx.notifier,
        ctx.balance,
        float(latest),
        ctx.config.get("telegram", {}).get("balance_updates", False),
    )
    return ctx.balance


def _ensure_ml_if_needed(cfg: dict) -> None:
    """Ensure ML model is loaded, preferring Supabase with fallback URL."""
    global _LAST_ML_CFG
    ml_cfg = {"ml_enabled": cfg.get("ml_enabled", True)}
    if ml_cfg != _LAST_ML_CFG:
        if not cfg.get("ml_enabled", True):
            _LAST_ML_CFG = ml_cfg
            return
        if not ML_AVAILABLE:
            symbol = cfg.get("symbol") or os.getenv("CT_SYMBOL", "XRPUSD")
            _available, reason = ml_utils.is_ml_available()
            logger.info(
                "ML model for %s unavailable (%s); ensure SUPABASE_URL and SUPABASE_KEY are set. "
                "Install cointrader-trainer only when training new models.",
                symbol,
                reason or "unknown reason",
            )
            cfg["ml_enabled"] = False
            ml_cfg["ml_enabled"] = False
            _LAST_ML_CFG = ml_cfg
            return
        symbol = cfg.get("symbol") or os.getenv("CT_SYMBOL", "XRPUSD")
        try:
            from crypto_bot.regime import regime_classifier as rc

            model, scaler, model_path = load_regime_model(symbol)
            if model is None:
                raise RuntimeError("regime model unavailable")
            rc._supabase_model = model
            rc._supabase_scaler = scaler
            rc._supabase_symbol = _norm_symbol(symbol)
            source = "Supabase"
            if model_path:
                path_str = str(model_path)
                if path_str.startswith("http"):
                    source = "URL"
                elif Path(path_str).exists():
                    source = "local file"
            logger.info(
                "Loaded global regime model for %s from %s: %s",
                symbol,
                source,
                model_path,
            )
        except Exception as exc:
            logger.warning("Supabase model load failed: %s", exc)
            fallback_url = (
                cfg.get("model_fallback_url")
                or os.getenv(
                    "CT_MODEL_FALLBACK_URL",
                    "https://prmhankbfjanqffwjcba.supabase.co/storage/v1/object/public/models/xrpusd_regime_lgbm.pkl",
                )
            )
            try:
                import urllib.request
                import pickle

                with urllib.request.urlopen(fallback_url) as resp:
                    model = pickle.loads(resp.read())
                rc._supabase_model = model
                rc._supabase_scaler = None
                rc._supabase_symbol = symbol
                logger.info(
                    "Loaded fallback regime model for %s from %s",
                    symbol,
                    fallback_url,
                )
            except Exception as url_exc:
                logger.warning(
                    "Failed to load fallback model from %s: %s", fallback_url, url_exc
                )
                logger.warning("ML disabled: %s", url_exc)
                cfg["ml_enabled"] = False
                ml_cfg["ml_enabled"] = False
        _LAST_ML_CFG = ml_cfg


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


def _load_config_file() -> dict:
    """Read and validate configuration from disk."""
    with open(CONFIG_PATH) as f:
        logger.info("Loading config from %s", CONFIG_PATH)
        data = yaml.safe_load(f) or {}
    trading_cfg = data.get("trading", {}) or {}
    raw_ex = data.get("exchange") or trading_cfg.get("exchange") or os.getenv("EXCHANGE")
    if isinstance(raw_ex, dict):
        exchange_cfg = dict(raw_ex)
    else:
        exchange_cfg = {"name": raw_ex}
    exchange_cfg.setdefault("name", "kraken")
    exchange_cfg.setdefault("max_concurrency", 3)
    exchange_cfg.setdefault("request_timeout_ms", 10000)
    exchange_id = exchange_cfg.get("name")
    timeframes = data.get("timeframes") or trading_cfg.get("timeframes")
    trading_mode = (
        data.get("execution_mode")
        or trading_cfg.get("mode")
        or "dry_run"
    )
    allowed_quotes = trading_cfg.get("allowed_quotes", [])
    logger.info(
        "Exchange=%s timeframes=%s mode=%s allowed_quotes=%s hft=%s",
        exchange_id,
        timeframes,
        trading_mode,
        allowed_quotes,
        trading_cfg.get("hft_enabled", False),
    )
    trading_cfg.setdefault("allowed_quotes", [])
    trading_cfg.setdefault("min_ticker_volume", 0)
    backfill_cfg = trading_cfg.get("backfill", {}) or {}
    backfill_cfg.setdefault("warmup_high_tf", [])
    backfill_cfg.setdefault("deep_low_tf", False)
    backfill_cfg.setdefault("deep_days", 0)
    trading_cfg["backfill"] = backfill_cfg
    trading_cfg.setdefault("hft_enabled", False)
    trading_cfg.setdefault("hft_symbols", [])
    trading_cfg.setdefault("exclude_symbols", [])
    trading_cfg.setdefault("require_sentiment", True)
    data["trading"] = trading_cfg
    data.setdefault("allowed_quotes", allowed_quotes)
    sf_cfg = data.get("symbol_filter", {}) or {}
    if allowed_quotes:
        sf_cfg.setdefault("quote_whitelist", allowed_quotes)
    data["symbol_filter"] = sf_cfg
    data["exchange"] = exchange_cfg
    data.setdefault("timeframes", timeframes)
    data.setdefault("execution_mode", trading_mode)

    risk_cfg = data.get("risk", {}) or {}
    data["risk"] = risk_cfg

    fees_cfg = data.get("fees", {}) or {}
    kraken_cfg = fees_cfg.get("kraken", {}) or {}
    kraken_cfg.setdefault("taker_bp", 0)
    kraken_cfg.setdefault("maker_bp", 0)
    fees_cfg["kraken"] = kraken_cfg
    data["fees"] = fees_cfg

    feat_cfg = data.get("features", {}) or {}
    feat_cfg.setdefault("ml", False)
    feat_cfg.setdefault("helius", False)
    feat_cfg.setdefault("pump_monitor", False)
    feat_cfg.setdefault("telegram", False)
    data["features"] = feat_cfg

    tele_cfg = data.get("telemetry", {}) or {}
    tele_cfg.setdefault("batch_summary_secs", 0)
    data["telemetry"] = tele_cfg

    # Default chunk size for OHLCV updates if not provided
    data.setdefault("ohlcv_chunk_size", 20)
    data_cfg = data.get("data", {}) or {}
    ml_cfg = data_cfg.get("market-loader", {}) or {}
    ml_cfg.setdefault("bootstrap_timeout_minutes", 10)
    data_cfg["market-loader"] = ml_cfg
    data["data"] = data_cfg
    data.setdefault("bootstrap_timeout_minutes", ml_cfg["bootstrap_timeout_minutes"])
    ohlcv_cfg = data.get("ohlcv", {}) or {}
    ohlcv_cfg.setdefault("bootstrap_timeframes", ["1h"])
    ohlcv_cfg.setdefault("defer_timeframes", ["4h", "1d"])
    data["ohlcv"] = ohlcv_cfg

    data = replace_placeholders(data)

    strat_dir = CONFIG_PATH.parent.parent / "config" / "strategies"
    trend_file = strat_dir / "trend_bot.yaml"
    if trend_file.exists():
        with open(trend_file) as sf:
            overrides = yaml.safe_load(sf) or {}
        trend_cfg = data.get("trend_bot", {})
        if isinstance(trend_cfg, dict):
            trend_cfg.update(overrides)
        else:
            trend_cfg = overrides
        data["trend_bot"] = trend_cfg

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
        if ScannerConfig is not None:
            if hasattr(ScannerConfig, "model_validate"):
                ScannerConfig.model_validate(data)
            else:  # pragma: no cover - for Pydantic < 2
                ScannerConfig.parse_obj(data)
    except ValidationError as exc:
        print("Invalid configuration:\n", exc)
        raise SystemExit(1)

    try:
        raw_scanner = data.get("solana_scanner", {}) or {}
        if SolanaScannerConfig is not None:
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
        if PythConfig is not None:
            if hasattr(PythConfig, "model_validate"):
                pyth_cfg = PythConfig.model_validate(raw_pyth)
            else:  # pragma: no cover - for Pydantic < 2
                pyth_cfg = PythConfig.parse_obj(raw_pyth)
            data["pyth"] = pyth_cfg.dict()
    except ValidationError as exc:
        print("Invalid configuration (pyth):\n", exc)
        raise SystemExit(1)

    return data


def _load_config_internal() -> tuple[dict, set[str]]:
    """Load config if underlying files changed and track updates."""
    global _CONFIG_CACHE

    main_stat = CONFIG_PATH.stat() if CONFIG_PATH.exists() else None
    main_mtime = main_stat.st_mtime if main_stat else 0.0
    main_size = main_stat.st_size if main_stat else 0
    strat_dir = CONFIG_PATH.parent.parent / "config" / "strategies"
    trend_file = strat_dir / "trend_bot.yaml"
    if trend_file.exists():
        trend_stat = trend_file.stat()
        trend_mtime = trend_stat.st_mtime
        trend_size = trend_stat.st_size
    else:
        trend_mtime = None
        trend_size = None

    if (
        _CONFIG_CACHE
        and _CONFIG_MTIMES.get(CONFIG_PATH) == (main_mtime, main_size)
        and _CONFIG_MTIMES.get(trend_file) == (trend_mtime, trend_size)
    ):
        return _CONFIG_CACHE, set()

    new_data = _load_config_file()
    changed = _diff_keys(_CONFIG_CACHE, new_data)
    _CONFIG_CACHE = new_data
    _CONFIG_MTIMES[CONFIG_PATH] = (main_mtime, main_size)
    if trend_mtime is not None:
        _CONFIG_MTIMES[trend_file] = (trend_mtime, trend_size)
    else:
        _CONFIG_MTIMES.pop(trend_file, None)

    _ensure_ml_if_needed(_CONFIG_CACHE)
    return _CONFIG_CACHE, changed


def load_config() -> dict:
    """Load YAML configuration for the bot synchronously."""
    cfg, _ = _load_config_internal()
    return cfg


async def load_config_async() -> tuple[dict, set[str]]:
    """Asynchronously load configuration returning changed sections."""
    return await asyncio.to_thread(_load_config_internal)


def maybe_reload_config(state: dict, config: dict) -> None:
    """Deprecated reload helper kept for backwards compatibility."""
    # Reloading now handled by :func:`reload_config` directly.
    return


def _diff_keys(old: dict, new: dict) -> set[str]:
    """Return top-level keys whose values differ between ``old`` and ``new``."""
    keys = set(old.keys()) | set(new.keys())
    return {k for k in keys if old.get(k) != new.get(k)}


def _merge_dict(dest: dict, src: dict) -> None:
    """Recursively merge ``src`` into ``dest`` in-place."""
    for key in list(dest.keys()):
        if key not in src:
            del dest[key]
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dest.get(key), dict):
            _merge_dict(dest[key], value)
        else:
            dest[key] = value


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def replace_placeholders(cfg):
    """Recursively replace ``${VAR}`` values with environment variables."""
    if isinstance(cfg, dict):
        return {k: replace_placeholders(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [replace_placeholders(v) for v in cfg]
    if isinstance(cfg, str):
        return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), m.group(0)), cfg)
    return cfg


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


async def reload_config(
    config: dict,
    ctx: BotContext,
    risk_manager: RiskManager,
    rotator: PortfolioRotator,
    position_guard: OpenPositionGuard,
    *,
    force: bool = False,
) -> None:
    """Reload configuration and update dependent components."""
    new_config, changed = await load_config_async()
    if not force and not changed:
        return

    _merge_dict(config, new_config)
    if (
        not config.get("symbols")
        and not config.get("onchain_symbols")
        and not config.get("symbol")
    ):
        config["symbol"] = fix_symbol(
            os.getenv("CT_SYMBOL", "XRP/USDT")
        )
    ctx.config = config
    new_hash = symbol_utils.compute_config_hash(config)
    old_hash = symbol_utils.get_cached_config_hash()
    if old_hash != new_hash:
        symbol_utils.invalidate_symbol_cache()
    symbol_utils._cached_hash = new_hash

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
    risk_manager.config = build_risk_config(config, volume_ratio)


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

    symbols = list(config.get("tradable_symbols", config.get("symbols", [])))
    top_n = int(config.get("scan_deep_top", 50))
    symbols = symbols[:top_n]
    if config.get("mode") != "cex":
        for sym in config.get("onchain_symbols", []):
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

    ohlcv_cfg = config.get("ohlcv", {})
    bootstrap_tfs = ohlcv_cfg.get(
        "bootstrap_timeframes", config.get("timeframes", ["1h"])
    )
    tfs = sf.get("initial_timeframes", bootstrap_tfs)
    tfs = sorted(
        set(tfs) | set(config.get("timeframes", [])),
        key=lambda t: timeframe_seconds(None, t),
    )
    tf_sec = timeframe_seconds(None, tfs[0])
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

    bootstrap_timeout = float(config.get("bootstrap_timeout_minutes", 10) or 10) * 60

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]

        async with symbol_cache_guard():
            async with OHLCV_LOCK:
                for tf in tfs:
                    logger.info("Starting OHLCV update for timeframe %s", tf)
                res = await _run_bootstrap(
                    update_multi_tf_ohlcv_cache,
                    exchange,
                    state.df_cache,
                    batch,
                    {**config, "timeframes": tfs},
                    limit=deep_limit,
                    start_since=history_since,
                    use_websocket=False,
                    force_websocket_history=config.get(
                        "force_websocket_history", False
                    ),
                    max_concurrent=config.get("max_concurrent_ohlcv"),
                    notifier=notifier,
                    priority_queue=symbol_priority_queue,
                    batch_size=ohlcv_batch_size,
                    run_timeout=config.get("bootstrap_timeout"),
                    timeout=bootstrap_timeout,
                )
                if res is not None:
                    state.df_cache = res

                res = await _run_bootstrap(
                    update_regime_tf_cache,
                    exchange,
                    state.regime_cache,
                    batch,
                    {**config, "timeframes": tfs},
                    limit=scan_limit,
                    start_since=lookback_since,
                    use_websocket=False,
                    force_websocket_history=config.get(
                        "force_websocket_history", False
                    ),
                    max_concurrent=config.get("max_concurrent_ohlcv"),
                    notifier=notifier,
                    df_map=state.df_cache,
                    batch_size=ohlcv_batch_size,
                    run_timeout=config.get("bootstrap_timeout"),
                    timeout=bootstrap_timeout,
                )
                if res is not None:
                    state.regime_cache = res
        logger.info("Deep historical OHLCV loaded for %d symbols", len(batch))

        processed += len(batch)
        pct = processed / total * 100
        logger.info("Initial scan %.1f%% complete", pct)
        if notifier and config.get("telegram", {}).get("status_updates", True):
            notifier.notify(f"Initial scan {pct:.1f}% complete")
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    register_task(
        asyncio.create_task(
            warm_deferred_timeframes(exchange, config, state, symbols)
        )
    )

    return


async def warm_deferred_timeframes(
    exchange: object,
    config: dict,
    state: "SessionState",
    symbols: list[str],
) -> None:
    """Warm OHLCV cache for deferred timeframes in the background."""

    ohlcv_cfg = config.get("ohlcv", {})
    defer_tfs = ohlcv_cfg.get("defer_timeframes")
    if not defer_tfs:
        return

    from crypto_bot.strategy import registry as strategy_registry

    strategies = strategy_registry.load_from_config(config)
    required_bars = strategy_registry.compute_required_lookback_per_tf(strategies)

    original_count = len(symbols)
    sf = config.get("symbol_filter", {})
    ohlcv_batch_size = config.get("ohlcv_batch_size")
    if ohlcv_batch_size is None:
        ohlcv_batch_size = sf.get("ohlcv_batch_size")
    scan_limit = int(
        sf.get("initial_history_candles", config.get("scan_lookback_limit", 50))
    )

    tfs = sorted(set(defer_tfs), key=lambda t: timeframe_seconds(None, t))
    regime_tfs = set(tf for tf in config.get("regime_timeframes", []) if tf in tfs)
    batch_size = int(config.get("symbol_batch_size", 10))
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        for tf in tfs:
            limit = max(int(required_bars.get(tf, 0)), scan_limit)
            tf_sec = timeframe_seconds(None, tf)
            lookback_since = int(time.time() * 1000 - limit * tf_sec * 1000)
            async with OHLCV_LOCK:
                state.df_cache = await update_multi_tf_ohlcv_cache(
                    exchange,
                    state.df_cache,
                    batch,
                    {**config, "timeframes": [tf]},
                    limit=limit,
                    start_since=lookback_since,
                    use_websocket=False,
                    force_websocket_history=config.get("force_websocket_history", False),
                    max_concurrent=config.get("max_concurrent_ohlcv"),
                    notifier=None,
                    priority_queue=symbol_priority_queue,
                    batch_size=ohlcv_batch_size,
                )

                if tf in regime_tfs:
                    state.regime_cache = await update_regime_tf_cache(
                        exchange,
                        state.regime_cache,
                        batch,
                        {**config, "regime_timeframes": [tf]},
                        limit=limit,
                        start_since=lookback_since,
                        use_websocket=False,
                        force_websocket_history=config.get(
                            "force_websocket_history", False
                        ),
                        max_concurrent=config.get("max_concurrent_ohlcv"),
                        notifier=None,
                        df_map=state.df_cache,
                        batch_size=ohlcv_batch_size,
                    )
    await asyncio.sleep(0)
    assert len(symbols) == original_count, "Symbol count changed during warm-up"

    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)


async def fetch_candidates(ctx: BotContext) -> None:
    """Gather symbols for this cycle and build the evaluation batch."""
    global symbol_priority_queue

    # Clean up any completed bootstrap tasks from previous cycles
    _prune_bootstrap_tasks()

    if not ctx.df_cache:
        no_data_symbols.clear()
    else:
        timeframe = ctx.config.get("timeframe", "1h")
        for sym in list(no_data_symbols):
            df = ctx.df_cache.get(timeframe, {}).get(sym)
            if isinstance(df, pd.DataFrame) and not df.empty:
                no_data_symbols.discard(sym)

    sf = ctx.config.setdefault("symbol_filter", {})

    if (
        not ctx.config.get("symbols")
        and not ctx.config.get("onchain_symbols")
        and not ctx.config.get("symbol")
    ):
        ctx.config["symbol"] = fix_symbol(
            os.getenv("CT_SYMBOL", "XRP/USDT")
        )
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
        scan_cfg = {**ctx.config}
        scan_cfg["symbols"] = ctx.config.get(
            "tradable_symbols", ctx.config.get("symbols", [])
        )
        symbols, _ = await get_filtered_symbols(ctx.exchange, scan_cfg)
    finally:
        if pump:
            sf["min_volume_usd"] = orig_min_volume
            sf["volume_percentile"] = orig_volume_pct
    symbols = [(s, sc) for s, sc in symbols if s not in no_data_symbols]
    allowed_syms = set(ctx.config.get("symbols", []))
    onchain_syms = [
        s
        for s in ctx.config.get("onchain_symbols", [])
        if s not in no_data_symbols and s in allowed_syms
    ]
    cex_candidates = list(symbols)
    onchain_candidates = [(s, 0.0) for s in onchain_syms]

    mode = ctx.config.get("mode", "auto")
    resolved_mode = mode
    if mode == "cex":
        active_candidates = cex_candidates
    elif mode == "onchain":
        active_candidates = onchain_candidates
    else:
        if not onchain_candidates:
            active_candidates = cex_candidates
            resolved_mode = "cex"
            if not getattr(fetch_candidates, "_onchain_fallback_logged", False):
                logger.info(
                    "Auto mode: falling back to CEX because 0 onchain candidates after metadata checks."
                )
                fetch_candidates._onchain_fallback_logged = True
        else:
            active_candidates = cex_candidates + onchain_candidates

    bases = [s.split("/")[0] for s in onchain_syms]
    meta_kept = 0
    meta_drop = 0
    if bases and resolved_mode != "cex":
        try:
            meta = await fetch_from_helius(bases)
            meta_kept = sum(1 for b in bases if b.upper() in meta)
            meta_drop = len(bases) - meta_kept
        except Exception as exc:  # pragma: no cover - metadata optional
            pipeline_logger.debug("metadata lookup failed: %s", exc)
            meta_kept = len(bases)
            meta_drop = 0
    pipeline_logger.info(
        "metadata_kept=%d metadata_dropped=%d (mode=%s)",
        meta_kept,
        meta_drop,
        ctx.config.get("mode"),
    )

    # Include benchmark pairs from configuration.
    default_symbol = ctx.config.get("symbol") or os.getenv("CT_SYMBOL")
    default_symbol = fix_symbol(default_symbol) if default_symbol else ""
    default_benchmarks = [default_symbol, "SOL/USDC"] if default_symbol else ["SOL/USDC"]
    benchmark_symbols = ctx.config.get("benchmark_symbols", default_benchmarks)
    if benchmark_symbols:
        active_candidates.extend((s, 10.0) for s in benchmark_symbols if s)

    symbols = active_candidates
    solana_tokens: list[str] = list(onchain_syms) if resolved_mode != "cex" else []
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

    if regime == "volatile" and resolved_mode != "cex":
        symbols.extend((s, 0.0) for s in onchain_syms)

    if regime == "volatile" and sol_cfg.get("enabled") and resolved_mode != "cex":
        try:
            new_tokens = await get_solana_new_tokens(sol_cfg)
            solana_tokens.extend(new_tokens)
            symbols.extend((m, 0.0) for m in new_tokens)
        except Exception as exc:  # pragma: no cover - best effort
            logger.error("Solana scanner failed: %s", exc)

    total_candidates = len(symbols)
    symbols = [(s, sc) for s, sc in symbols if s not in no_data_symbols]
    allowed_syms = set(ctx.config.get("symbols", []))
    allowed_syms.update(benchmark_symbols or [])
    symbols = [(s, sc) for s, sc in symbols if s in allowed_syms]
    ctx.active_universe = [s for s, _ in symbols]
    ctx.resolved_mode = resolved_mode

    logger.info(
        "Symbol summary: total=%d selected=%d filtered=%d first=%s",
        total_candidates,
        len(symbols),
        total_candidates - len(symbols),
        [s for s, _ in symbols[:5]],
    )

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

    fs_cfg = ctx.config.get("runtime", {}).get("fast_start", {})
    base_size_cfg = ctx.config.get("symbol_batch_size", 10)
    async with QUEUE_LOCK:
        if fs_cfg.get("enabled"):
            follow_size = fs_cfg.get("followup_batch_size", base_size_cfg)
            if not symbol_priority_queue:
                seeds = symbol_utils.select_seed_symbols(
                    symbols, ctx.exchange, ctx.config
                )
                rest_scores = [(s, sc) for s, sc in symbols if s not in seeds]
                rest_queue = list(build_priority_queue(rest_scores))
                ctx._fast_start_batches = [
                    rest_queue[i : i + follow_size]
                    for i in range(0, len(rest_queue), follow_size)
                ]
                symbol_priority_queue = deque(seeds)
                for sym in seeds:
                    logger.info("OHLCV[1m] warmup met for %s → enqueue", sym)
                base_size = len(seeds) if seeds else follow_size
            else:
                if len(symbol_priority_queue) < follow_size and getattr(
                    ctx, "_fast_start_batches", []
                ):
                    symbol_priority_queue.extend(ctx._fast_start_batches.pop(0))
                base_size = follow_size
        else:
            if not symbol_priority_queue:
                from crypto_bot.utils.symbol_pre_filter import liq_cache

                selected_symbols = [s for s, _ in symbols]
                volume_24h = {
                    s: (liq_cache.get(s) or (0.0, 0.0, 0.0))[0]
                    for s in selected_symbols
                }
                liquid = sorted(
                    selected_symbols,
                    key=lambda s: volume_24h.get(s, 0.0),
                    reverse=True,
                )[:150]
                liquid_scores = [(s, sc) for s, sc in symbols if s in liquid]
                rest_scores = [(s, sc) for s, sc in symbols if s not in liquid]
                symbol_priority_queue = build_priority_queue(liquid_scores)
                symbol_priority_queue.extend(build_priority_queue(rest_scores))
            base_size = base_size_cfg

        if onchain_syms and resolved_mode != "cex":
            for sym in reversed(onchain_syms):
                if sym not in symbol_priority_queue:
                    symbol_priority_queue.appendleft(sym)
        if solana_tokens:
            enqueue_solana_tokens(solana_tokens)

        batch_size = int(base_size * volatility_factor)
        if len(symbol_priority_queue) < batch_size and (
            not fs_cfg.get("enabled") or not getattr(ctx, "_fast_start_batches", [])
        ):
            symbol_priority_queue.extend(build_priority_queue(symbols))

        # Remove duplicates while preserving order
        symbol_priority_queue = deque(dict.fromkeys(symbol_priority_queue))

        # Keep only exchange-listed pairs if available
        if hasattr(ctx.exchange, "list_markets"):
            markets = ctx.exchange.list_markets()
            if isinstance(markets, dict):
                listed = set(markets)
            else:
                listed = set(markets or [])
            queue_list = list(symbol_priority_queue)
            symbol_priority_queue = deque(s for s in queue_list if s in listed)
            dropped = [s for s in queue_list if s not in listed]
            for sym in dropped:
                logger.info("Dropped non-exchange pair %s", sym)

        ctx.current_batch = [
            symbol_priority_queue.popleft()
            for _ in range(min(batch_size, len(symbol_priority_queue)))
        ]

        for sym in ctx.current_batch:
            if sym in recent_solana_set:
                recent_solana_set.discard(sym)
                try:
                    recent_solana_tokens.remove(sym)
                except ValueError:
                    pass

    logger.info("Current batch: %s", ctx.current_batch)


async def scan_arbitrage(exchange: object, config: dict) -> list[str]:
    """Return symbols with profitable Solana arbitrage opportunities."""
    pairs: list[str] = config.get("arbitrage_pairs", [])
    if not pairs:
        return []

    gecko_prices: dict[str, float] = {}
    if fetch_geckoterminal_ohlcv:
        for sym in pairs:
            try:
                data = await fetch_geckoterminal_ohlcv(sym, limit=1)
            except Exception:
                data = None
            if data:
                # close price of latest candle
                gecko_prices[sym] = data[-1][4]

    remaining = [s for s in pairs if s not in gecko_prices]
    dex_prices: dict[str, float] = gecko_prices.copy()
    if remaining and fetch_solana_prices:
        dex_prices.update(await fetch_solana_prices(remaining))
    results: list[str] = []
    threshold = float(
        config.get(
            "arbitrage_threshold",
            config.get("grid_bot", {}).get("arbitrage_threshold", 0.0),
        )
    )

    tickers: dict[str, dict[str, Any]] = {}
    try:
        has = getattr(exchange, "has", {})
        if getattr(has, "get", lambda _k, _d=None: _d)("fetchTickers"):
            try:
                if asyncio.iscoroutinefunction(
                    getattr(exchange, "fetch_tickers", None)
                ):
                    tickers = await exchange.fetch_tickers(pairs)
                else:
                    tickers = await asyncio.to_thread(exchange.fetch_tickers, pairs)
            except Exception:
                tickers = {}
    except Exception:
        tickers = {}

    for sym in pairs:
        dex_price = dex_prices.get(sym)
        if not dex_price:
            continue
        ticker = tickers.get(sym)
        if ticker is None:
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
    primary_tickers: dict[str, dict] = {}
    secondary_tickers: dict[str, dict] = {}

    if getattr(primary, "has", {}).get("fetchTickers") and hasattr(primary, "fetch_tickers"):
        try:
            if asyncio.iscoroutinefunction(getattr(primary, "fetch_tickers", None)):
                primary_tickers = await primary.fetch_tickers(pairs)
            else:
                primary_tickers = await asyncio.to_thread(primary.fetch_tickers, pairs)
        except Exception:
            primary_tickers = {}

    if getattr(secondary, "has", {}).get("fetchTickers") and hasattr(secondary, "fetch_tickers"):
        try:
            if asyncio.iscoroutinefunction(getattr(secondary, "fetch_tickers", None)):
                secondary_tickers = await secondary.fetch_tickers(pairs)
            else:
                secondary_tickers = await asyncio.to_thread(secondary.fetch_tickers, pairs)
        except Exception:
            secondary_tickers = {}

    for sym in pairs:
        t1 = primary_tickers.get(sym)
        if t1 is None:
            try:
                if asyncio.iscoroutinefunction(getattr(primary, "fetch_ticker", None)):
                    t1 = await primary.fetch_ticker(sym)
                else:
                    t1 = await asyncio.to_thread(primary.fetch_ticker, sym)
            except Exception:
                continue

        t2 = secondary_tickers.get(sym)
        if t2 is None:
            try:
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


async def update_caches(ctx: BotContext, chunk_size: int | None = None) -> None:
    async with symbol_cache_guard():
        await _update_caches_impl(ctx, chunk_size)


async def _update_caches_impl(ctx: BotContext, chunk_size: int | None = None) -> None:
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

    ohlcv_chunk_size = ctx.config.get("ohlcv_chunk_size")
    if ohlcv_chunk_size is None:
        ohlcv_chunk_size = 20

    max_concurrent = ctx.config.get("max_concurrent_ohlcv")
    if isinstance(max_concurrent, (int, float)):
        max_concurrent = int(max_concurrent)
        if ctx.volatility_factor > 5:
            max_concurrent = max(1, max_concurrent // 2)

    tfs = sorted(
        set(ctx.config.get("timeframes", [])),
        key=lambda t: timeframe_seconds(None, t),
    )

    bootstrap_timeout = float(ctx.config.get("bootstrap_timeout_minutes", 10) or 10) * 60

    async with OHLCV_LOCK:
        try:
            for tf in tfs:
                logger.info("Starting OHLCV update for timeframe %s", tf)
            ctx.df_cache = await update_multi_tf_ohlcv_cache(
                ctx.exchange,
                ctx.df_cache,
                batch,
                {**ctx.config, "timeframes": tfs},
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
                timeout=bootstrap_timeout,
            )
        except Exception as exc:
            logger.warning("WS OHLCV failed: %s - falling back to REST", exc)
            for tf in tfs:
                logger.info("Starting OHLCV update for timeframe %s", tf)
            ctx.df_cache = await update_multi_tf_ohlcv_cache(
                ctx.exchange,
                ctx.df_cache,
                batch,
                {**ctx.config, "timeframes": tfs},
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
                timeout=bootstrap_timeout,
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
            timeout=bootstrap_timeout,
        )

    filtered_batch: list[str] = []
    for sym in batch:
        df = ctx.df_cache.get(timeframe, {}).get(sym)
        count = len(df) if isinstance(df, pd.DataFrame) else 0
        logger.info("%s OHLCV: %d candles", sym, count)
        if count == 0:
            logger.warning("No OHLCV data for %s; skipping analysis", sym)
            async with QUEUE_LOCK:
                try:
                    symbol_priority_queue.remove(sym)
                except ValueError:
                    pass
            no_data_symbols.add(sym)
            continue
        no_data_symbols.discard(sym)
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

    if ctx.config.get("use_websocket", True) and ctx.current_batch:
        timeframe = ctx.config.get("timeframe", "1h")

        async def subscribe(sym: str) -> None:
            try:
                params = inspect.signature(ctx.exchange.watchOHLCV).parameters
                kwargs = {"symbol": sym, "timeframe": timeframe}
                if "timeout" in params:
                    kwargs["timeout"] = WS_OHLCV_TIMEOUT
                await ctx.exchange.watchOHLCV(**kwargs)
            except Exception as exc:  # pragma: no cover - network
                logger.warning("WS subscribe failed for %s: %s", sym, exc)

        await asyncio.gather(*(subscribe(sym) for sym in ctx.current_batch))

    ctx.timing["ohlcv_fetch_latency"] = time.perf_counter() - start


async def enrich_with_pyth(ctx: BotContext) -> None:
    """Update cached OHLCV using the latest Pyth prices."""
    batch = ctx.current_batch
    if not batch:
        return
    async with symbol_cache_guard():
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
                    if (
                        attrs.get("base") == base
                        and attrs.get("quote_currency") == "USD"
                    ):
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
    async with symbol_cache_guard():
        await _analyse_batch_impl(ctx)


async def _analyse_batch_impl(ctx: BotContext) -> None:
    """Run signal analysis on the current batch."""
    batch = ctx.current_batch
    trading_cfg = ctx.config.get("trading", {})
    hft_enabled = bool(trading_cfg.get("hft_enabled", False))
    hft_symbols = set(trading_cfg.get("hft_symbols", []))

    allowed_quotes = set(
        trading_cfg.get("allowed_quotes")
        or ctx.config.get("allowed_quote_currencies", [])
    )
    exclude_symbols = set(
        trading_cfg.get("exclude_symbols", ctx.config.get("excluded_symbols", []))
    )

    # Apply symbol filters before evaluation
    filtered: list[str] = []
    for sym in batch:
        base, _, quote = sym.partition("/")
        if allowed_quotes and quote not in allowed_quotes:
            continue
        if sym in exclude_symbols:
            continue
        filtered.append(sym)
    batch = filtered

    logger.info(
        "Strategy evaluation starting with %d symbols (mode=%s, hft=%s)",
        len(batch),
        getattr(ctx, "resolved_mode", ctx.config.get("mode", "auto")),
        hft_enabled,
    )
    if not batch:
        logger.info("analyse_batch called with empty batch")
        ctx.analysis_results = []
        return

    logger.info(
        "analyse_batch starting with %d symbols: %s",
        len(batch),
        batch,
    )
    pipeline_logger.info(
        "strategy_eval_start=%d timeframes=%s",
        len(batch),
        sorted(ctx.df_cache.keys()),
    )

    base_tf = ctx.config.get("timeframe", "1h")
    try:
        base_minutes = int(pd.Timedelta(base_tf).total_seconds() // 60)
    except Exception:  # pragma: no cover - invalid timeframe
        base_minutes = 0
    ctx.analysis_errors = 0
    ctx.analysis_timeouts = 0

    eval_cfg = ctx.config.get("runtime", {}).get("evaluation", {})
    timeout = eval_cfg.get("per_symbol_timeout_s")
    mode = ctx.config.get("mode", "cex")

    async def eval_fn(symbol: str) -> Any:
        df_map = {tf: c.get(symbol) for tf, c in ctx.df_cache.items()}
        for tf, cache in ctx.regime_cache.items():
            df_map[tf] = cache.get(symbol)
        df = df_map.get(base_tf)

        if hft_enabled and (base_minutes <= 1 or symbol in hft_symbols):
            from crypto_bot.hft import HFTEngine, maker_spread

            engine = getattr(ctx, "hft_engine", None)
            if engine is None:
                engine = HFTEngine()
                ctx.hft_engine = engine
            engine.attach(symbol, maker_spread)
            return {"symbol": symbol, "skip": True}

        logger.info(
            "DF len for %s: %d",
            symbol,
            len(df) if isinstance(df, pd.DataFrame) else 0,
        )
        try:
            coro = analyze_symbol(
                symbol,
                df_map,
                mode,
                ctx.config,
                ctx.notifier,
                mempool_monitor=ctx.mempool_monitor,
                mempool_cfg=ctx.mempool_cfg,
            )
            if timeout:
                return await asyncio.wait_for(coro, timeout)
            return await coro
        except asyncio.TimeoutError:
            logger.error("Analysis timeout for %s", symbol)
            return {"symbol": symbol, "skip": True}

    ctx.eval_fn = eval_fn
    from crypto_bot.evaluator import evaluate_batch

    results_map = await evaluate_batch(batch, ctx)

    ctx.analysis_results = [
        res for sym in batch if (res := results_map.get(sym)) is not None
    ]

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
    pipeline_logger.info(
        "strategy_eval_done signals=%d",
        sum(1 for r in ctx.analysis_results if not r.get("skip")),
    )
    try:
        from crypto_bot.utils.telemetry import dump as telemetry_dump

        telemetry_dump()
    except Exception:  # pragma: no cover - telemetry optional
        pass


async def execute_signals(ctx: BotContext) -> None:
    """Open trades for qualified analysis results."""
    from collections import Counter

    results = getattr(ctx, "analysis_results", [])
    if not results:
        logger.info("No analysis results to act on")
        return
    if "pipeline_logger" not in globals():
        globals()["pipeline_logger"] = logger
    if "state" not in globals():
        globals()["state"] = {"running": True}
    # Log raw analysis results before filtering to clarify decision paths
    min_score = ctx.config.get("yamlrouter", {}).get("min_score", 0.0)
    for res in results:
        sym = res.get("symbol", "")
        score = float(res.get("score", 0.0))
        direction = res.get("direction", "none")
        atr = res.get("atr")
        logger.info(
            "Raw result: symbol=%s score=%.2f direction=%s atr=%s",
            sym,
            score,
            direction,
            atr,
        )
        reasons: list[str] = []
        if score <= min_score:
            reasons.append("score below min_score")
        if direction == "none":
            reasons.append("no direction")
        if res.get("too_flat", False):
            reasons.append("atr too flat")
        if direction == "short" and not ctx.config.get("allow_short", True):
            reasons.append("short selling disabled")
        if reasons:
            logger.warning("Skipping %s: %s", sym, ", ".join(reasons))
            continue
        if ctx.config.get("execution_mode") == "dry_run":
            logger.info("Executing dry-run trade...")
            try:
                await cex_trade_async(
                    ctx.exchange,
                    ctx.ws_client,
                    sym,
                    direction_to_side(direction),
                    0.0,
                    ctx.notifier,
                    dry_run=True,
                    config=ctx.config.get("exec"),
                    score=score,
                    reason="pre-filter",
                )
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.warning("Dry-run trade failed for %s: %s", sym, exc)
    ctx.reject_reasons = {}
    reject_counts = Counter()

    def _log_rejection(
        sym: str, score: float, direction: str, min_score: float, verdict: str, category: str
    ) -> None:
        """Emit structured log for a rejected candidate."""
        score_logger.info(
            "[REJECT][%s] sym=%s score=%.2f dir=%s min_score=%.2f verdict=%s",
            category,
            sym,
            score,
            direction,
            min_score,
            verdict,
        )

    # Filter and prioritize by score
    orig_results = results
    results = []
    skipped_syms: list[str] = []
    no_dir_syms: list[str] = []
    low_score: list[str] = []
    dry_run = ctx.config.get("execution_mode") == "dry_run"
    for r in orig_results:
        logger.debug("Analysis result: %s", r)
        sym = r.get("symbol", "")
        direction = r.get("direction", "none")
        min_req = r.get("min_confidence", ctx.config.get("min_confidence_score", 0.0))
        score = r.get("score", 0.0)
        if r.get("skip"):
            skipped_syms.append(sym)
            _log_rejection(sym, score, direction, min_req, "skip_flag", "SCORING")
            reject_counts["skip_flag"] += 1
            continue
        if direction == "none":
            no_dir_syms.append(sym)
            _log_rejection(sym, score, direction, min_req, "no_direction", "SCORING")
            reject_counts["no_direction"] += 1
            continue
        if score < min_req:
            low_score.append(f"{sym}({score:.2f}<{min_req:.2f})")
            _log_rejection(sym, score, direction, min_req, "below_min_score", "SCORING")
            reject_counts["below_min_score"] += 1
            continue
        score_logger.info(
            "Passing to execute: symbol=%s, score=%s, dry_run=%s",
            sym,
            score,
            dry_run,
        )
        results.append(r)

    score_logger.debug(
        "Candidate scoring: %d/%d met the minimum score; low_score=%s skip=%s no_direction=%s",
        len(results),
        len(orig_results),
        low_score,
        skipped_syms,
        no_dir_syms,
    )

    if not results:
        score_logger.info("All signals filtered out - nothing actionable")
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
        min_req = candidate.get(
            "min_confidence", ctx.config.get("min_confidence_score", 0.0)
        )
        score = candidate.get("score", 0.0)
        direction = candidate.get("direction", "none")
        if ctx.position_guard and not ctx.position_guard.can_open(ctx.positions):
            logger.debug("Position guard blocked opening a new position")
            _log_rejection(
                candidate.get("symbol", ""),
                score,
                direction,
                min_req,
                "max_open_trades",
                "POSITION_GUARD",
            )
            reject_counts["position_guard_limit"] += 1
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
            _log_rejection(sym, score, direction, min_req, outcome_reason, "POSITION_GUARD")
            logger.info("[EVAL] %s -> %s", sym, outcome_reason)
            reject_counts["existing_position"] += 1
            continue

        df = candidate["df"]
        price = df["close"].iloc[-1]
        strategy = candidate.get("name", "")
        if candidate.get("too_flat", False):
            outcome_reason = "atr too flat"
            _log_rejection(sym, score, direction, min_req, outcome_reason, "RISK_MANAGER")
            logger.info("[EVAL] %s -> %s", sym, outcome_reason)
            reject_counts["atr_too_flat"] += 1
            continue
        allowed, reason = ctx.risk_manager.allow_trade(df, strategy, sym)
        if not allowed:
            outcome_reason = f"blocked: {reason}"
            _log_rejection(sym, score, direction, min_req, reason, "RISK_MANAGER")
            logger.info("[EVAL] %s -> %s", sym, outcome_reason)
            key = reason.lower().replace(" ", "_")
            reject_counts[key] += 1
            continue

        sentiment_factor = ctx.risk_manager.sentiment_factor_or_default(
            ctx.config.get("trading", {}).get("require_sentiment", True)
        )

        probs = candidate.get("probabilities", {})
        reg_prob = float(probs.get(candidate.get("regime"), 0.0))
        size = (
            ctx.risk_manager.position_size(
                reg_prob,
                ctx.balance,
                df,
                atr=candidate.get("atr"),
                price=price,
                name=strategy,
                direction=direction,
            )
            * sentiment_factor
        )
        if size == 0:
            await refresh_balance(ctx)
            size = (
                ctx.risk_manager.position_size(
                    reg_prob,
                    ctx.balance,
                    df,
                    atr=candidate.get("atr"),
                    price=price,
                    name=strategy,
                    direction=direction,
                )
                * sentiment_factor
            )
            if size == 0:
                outcome_reason = f"size {size:.4f}"
                _log_rejection(sym, score, direction, min_req, outcome_reason, "RISK_MANAGER")
                logger.info("[EVAL] %s -> %s", sym, outcome_reason)
                reject_counts["size_zero"] += 1
                continue

        if not ctx.risk_manager.can_allocate(strategy, abs(size), ctx.balance):
            logger.info(
                "Insufficient capital to allocate %.4f for %s via %s",
                size,
                sym,
                strategy,
            )
            outcome_reason = "insufficient capital"
            _log_rejection(sym, score, direction, min_req, outcome_reason, "RISK_MANAGER")
            logger.info("[EVAL] %s -> %s", sym, outcome_reason)
            reject_counts["insufficient_capital"] += 1
            continue

        amount = abs(size) / price if price > 0 else 0.0
        side = direction_to_side(direction)
        if side == "sell" and not ctx.config.get("allow_short", False):
            outcome_reason = "short selling disabled"
            _log_rejection(sym, score, direction, min_req, outcome_reason, "RISK_MANAGER")
            logger.info("[EVAL] %s -> %s", sym, outcome_reason)
            reject_counts["short_selling_disabled"] += 1
            continue
        start_exec = time.perf_counter()
        executed_via_sniper = False
        executed_via_cross = False
        mode_str = (
            "dry_run" if ctx.config.get("execution_mode") == "dry_run" else "live"
        )
        trade_reason = candidate.get("reason") or strategy
        if strategy == "cross_chain_arb_bot":
            logger.info(
                "TRADE (%s) %s %s qty=%.4f price=spot reason='%s'",
                mode_str,
                side.upper(),
                sym,
                amount,
                trade_reason,
            )
            task = register_task(
                asyncio.create_task(
                    cross_chain_trade(
                        ctx.exchange,
                        ctx.ws_client,
                        sym,
                        side,
                        amount,
                        dry_run=ctx.config.get("execution_mode") == "dry_run",
                        slippage_bps=ctx.config.get("solana_slippage_bps", 50),
                        use_websocket=ctx.config.get("use_websocket", False),
                        notifier=ctx.notifier,
                        mempool_monitor=ctx.mempool_monitor,
                        mempool_cfg=ctx.mempool_cfg,
                        config=ctx.config,
                    )
                )
            )
            CROSS_ARB_TASKS.add(task)
            task.add_done_callback(CROSS_ARB_TASKS.discard)
            executed_via_cross = True
        elif sym.endswith("/USDC"):
            if sym in NEW_SOLANA_TOKENS:
                reg = candidate.get("regime")
                if reg not in {"volatile", "breakout"}:
                    logger.info("[EVAL] %s -> regime %s not tradable", sym, reg)
                    NEW_SOLANA_TOKENS.discard(sym)
                    continue

            sol_score, _ = sniper_solana.generate_signal(
                df,
                symbol=sym,
                timeframe=ctx.config.get("timeframe"),
            )
            if sym in NEW_SOLANA_TOKENS:
                NEW_SOLANA_TOKENS.discard(sym)
            if sol_score > 0.7:
                base, quote = sym.split("/")
                logger.info(
                    "TRADE (%s) %s %s qty=%.4f price=spot reason='%s'",
                    mode_str,
                    side.upper(),
                    sym,
                    amount,
                    trade_reason,
                )
                task = register_task(
                    asyncio.create_task(
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
                )
                SNIPER_TASKS.add(task)
                task.add_done_callback(SNIPER_TASKS.discard)
                executed_via_sniper = True

        if not executed_via_sniper and not executed_via_cross:
            logger.info(
                "TRADE (%s) %s %s qty=%.4f price=spot reason='%s'",
                mode_str,
                side.upper(),
                sym,
                amount,
                trade_reason,
            )
            logger.info(
                f"Attempting trade for {candidate['symbol']}: score={candidate.get('score')}, direction={candidate.get('direction')}, after filters"
            )
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
                score=candidate.get("score", 0.0),
                reason=trade_reason,
                trading_paused=not state.get("running", True),
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

        if ctx.config.get("execution_mode") == "dry_run" and ctx.wallet:
            try:
                if side == "buy":
                    ctx.wallet.buy(sym, amount, price)
                else:
                    ctx.wallet.sell(sym, amount, price)
                ctx.balance = ctx.wallet.total_balance
            except Exception:
                pass
        ctx.risk_manager.allocate_capital(strategy, abs(size))
        if ctx.config.get("execution_mode") == "dry_run":
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
        else:
            try:
                pos_list = await cex_trade_async.sync_positions_async(ctx.exchange)
                ctx.positions = {p.get("symbol"): p for p in pos_list}
            except Exception as exc:  # pragma: no cover - optional
                logger.error("Position sync failed: %s", exc)
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

        if strategy == "micro_scalp_bot":
            register_task(asyncio.create_task(_monitor_micro_scalp_exit(ctx, sym)))

    if executed == 0:
        logger.info(
            "No trades executed from %d candidate signals", len(results[:top_n])
        )

    if reject_counts:
        ctx.reject_reasons = dict(reject_counts)
        top = ", ".join(f"{k}={v}" for k, v in reject_counts.most_common())
        pipeline_logger.info("Reject reasons (top): %s", top)


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
        dca_score, dca_dir = dca_bot.generate_signal(
            df, symbol=sym, timeframe=tf
        )
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
                    score=0.0,
                    reason=pos.get("strategy", ""),
                    trading_paused=not state.get("running", True),
                )
                if ctx.config.get("execution_mode") == "dry_run" and ctx.wallet:
                    try:
                        if pos["side"] == "buy":
                            ctx.wallet.buy(sym, add_amount, current_price)
                        else:
                            ctx.wallet.sell(sym, add_amount, current_price)
                        ctx.balance = ctx.wallet.total_balance
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
                score=0.0,
                reason=pos.get("strategy", ""),
                trading_paused=not state.get("running", True),
            )
            if ctx.config.get("execution_mode") == "dry_run" and ctx.wallet:
                try:
                    if pos["side"] == "buy":
                        ctx.wallet.sell(sym, pos["size"], current_price)
                    else:
                        ctx.wallet.buy(sym, pos["size"], current_price)
                    ctx.balance = ctx.wallet.total_balance
                except Exception:
                    pass
            await refresh_balance(ctx)
            realized_pnl = (current_price - pos["entry_price"]) * pos["size"]
            if pos["side"] == "sell":
                realized_pnl = -realized_pnl
            pnl_logger.log_pnl(
                pos.get("regime", ""),
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
            score, direction = dca_bot.generate_signal(
                df, symbol=sym, timeframe=tf
            )
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
                    score=0.0,
                    reason=pos.get("strategy", ""),
                    trading_paused=not state.get("running", True),
                )
                if ctx.config.get("execution_mode") == "dry_run" and ctx.wallet:
                    try:
                        if pos["side"] == "buy":
                            ctx.wallet.buy(sym, new_size, current_price)
                        else:
                            ctx.wallet.sell(sym, new_size, current_price)
                        ctx.balance = ctx.wallet.total_balance
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
            score=0.0,
            reason=pos.get("strategy", ""),
            trading_paused=not state.get("running", True),
        )

        if ctx.config.get("execution_mode") == "dry_run" and ctx.wallet:
            try:
                if pos["side"] == "buy":
                    ctx.wallet.sell(sym, pos["size"], exit_price)
                else:
                    ctx.wallet.buy(sym, pos["size"], exit_price)
                ctx.balance = ctx.wallet.total_balance
            except Exception:
                pass

        await refresh_balance(ctx)
        realized_pnl = (exit_price - pos["entry_price"]) * pos["size"]
        if pos["side"] == "sell":
            realized_pnl = -realized_pnl
        pnl_logger.log_pnl(
            pos.get("regime", ""),
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

    if not ctx.positions and ctx.wallet and ctx.wallet.positions:
        for pid, wpos in list(ctx.wallet.positions.items()):
            sym = wpos.get("symbol") or pid
            df = tf_cache.get(sym)
            exit_price = wpos.get("entry_price", 0.0)
            if df is not None and not df.empty:
                exit_price = float(df["close"].iloc[-1])

            size = wpos.get("size", wpos.get("amount", 0.0))
            try:
                if wpos.get("side") == "buy":
                    ctx.wallet.sell(sym, size, exit_price)
                else:
                    ctx.wallet.buy(sym, size, exit_price)
                ctx.balance = ctx.wallet.total_balance
            except Exception:
                pass
            realized_pnl = (exit_price - wpos.get("entry_price", 0.0)) * size
            if wpos.get("side") == "sell":
                realized_pnl = -realized_pnl
            pnl_logger.log_pnl(
                wpos.get("regime", ""),
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

        if ctx.wallet:
            ctx.wallet.positions.clear()


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
        score=0.0,
        reason=pos.get("strategy", ""),
        trading_paused=not state.get("running", True),
    )

    if ctx.config.get("execution_mode") == "dry_run" and ctx.wallet:
        try:
            if pos["side"] == "buy":
                ctx.wallet.sell(sym, pos["size"], exit_price)
            else:
                ctx.wallet.buy(sym, pos["size"], exit_price)
            ctx.balance = ctx.wallet.total_balance
        except Exception:
            pass

    await refresh_balance(ctx)
    realized_pnl = (exit_price - pos["entry_price"]) * pos["size"]
    if pos["side"] == "sell":
        realized_pnl = -realized_pnl
    pnl_logger.log_pnl(
        pos.get("regime", ""),
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


async def _main_impl() -> MainResult:
    """Implementation for running the trading bot."""
    logger.info("Starting bot")
    global UNKNOWN_COUNT, TOTAL_ANALYSES
    config, _ = await load_config_async()
    config["timeframes"] = collect_timeframes(config)
    if (
        not config.get("symbols")
        and not config.get("onchain_symbols")
        and not config.get("symbol")
    ):
        config["symbol"] = fix_symbol(
            os.getenv("CT_SYMBOL", "XRP/USDT")
        )
    env_chunk = os.getenv("OHLCV_CHUNK_SIZE")
    if env_chunk:
        try:
            config["ohlcv_batch_size"] = int(env_chunk)
        except ValueError:
            logger.warning("Invalid OHLCV_CHUNK_SIZE %r", env_chunk)
    stop_reason = "completed"

    mapping = await load_token_mints()
    if mapping:
        set_token_mints({**TOKEN_MINTS, **mapping})
    onchain_syms = [fix_symbol(s) for s in config.get("onchain_symbols", [])]
    onchain_syms = [f"{s}/USDC" if "/" not in s else s for s in onchain_syms]
    sol_syms = [fix_symbol(s) for s in config.get("solana_symbols", [])]
    sol_syms = [f"{s}/USDC" if "/" not in s else s for s in sol_syms]
    if sol_syms:
        config["solana_symbols"] = sol_syms
    merged_syms = list(dict.fromkeys(onchain_syms + sol_syms))
    if merged_syms:
        config["onchain_symbols"] = merged_syms
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
                        enqueue_solana_tokens(tokens)
                        for sym in reversed(tokens):
                            symbol_priority_queue.appendleft(sym)
                            NEW_SOLANA_TOKENS.add(sym)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - best effort
                logger.error("Solana scan error: %s", exc)
            await wait_or_event(interval)

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
    secrets = {k: _norm(v) for k, v in (dotenv_values(ENV_PATH) or {}).items()}
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

    # Export secrets into the process env for downstream libs (skip None, coerce to str)
    os.environ.update({k: str(v) for k, v in (secrets or {}).items() if v is not None})

    # Validate required environment variables for optional features
    if config.get("ml_enabled", True) and not os.environ.get("SUPABASE_URL"):
        config["ml_enabled"] = False
        config.setdefault("features", {})["ml"] = False
        logger.warning("SUPABASE_URL missing; disabling ML features")

    user = load_or_create(interactive=False)

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
    global TASK_FAILURE_NOTIFIER
    TASK_FAILURE_NOTIFIER = notifier
    if status_updates:
        notifier.notify("🤖 CoinTrader2.0 started")

    mempool_cfg = config.get("mempool_monitor", {})
    mempool_monitor = None
    if mempool_cfg.get("enabled"):
        mempool_monitor = SolanaMempoolMonitor()

    wake_event = asyncio.Event()

    async def wait_or_event(timeout: float) -> None:
        """Wait for an external event or until ``timeout`` elapses."""
        end = asyncio.get_running_loop().time() + timeout
        while True:
            wake_event.clear()
            remaining = end - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(wake_event.wait(), timeout=min(1.0, remaining))
                break
            except asyncio.TimeoutError:
                continue

    if notifier.token and notifier.chat_id:
        try:
            await send_test_message(notifier.token, notifier.chat_id, "Bot started")
        except Exception:
            logger.warning("Telegram test message failed; check your token and chat ID")

    # allow user-configured exchange to override YAML setting
    if user.get("exchange"):
        config["primary_exchange"] = user["exchange"]

    if config.get("exchanges"):
        exchanges = get_exchanges(config)
        ex_name = config.get("exchange")
        if isinstance(ex_name, dict):
            ex_name = ex_name.get("name")
        primary = config.get("primary_exchange") or ex_name or next(iter(exchanges))
        exchange, ws_client = exchanges[primary]
        secondary_exchange = None
        for name, pair in exchanges.items():
            if name != primary:
                secondary_exchange = pair[0]
                break
    else:
        exchange, ws_client = get_exchange(config)
        secondary_exchange = None
    if hasattr(exchange, "options"):
        opts = getattr(exchange, "options", {})
        opts["ws"] = {"ping_interval": 10, "ping_timeout": 45}
        exchange.options = opts

    ping_interval = int(config.get("ws_ping_interval", 0) or 0)
    if ping_interval > 0 and hasattr(exchange, "ping"):
        task = register_task(
            asyncio.create_task(_ws_ping_loop(exchange, ping_interval))
        )
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
            await wait_or_event(delay)
            delay = min(delay * 2, MAX_SYMBOL_SCAN_DELAY)

        if discovered:
            sf_cfg = config.get("symbol_filter", {})
            tradable = await build_tradable_set(
                exchange,
                allowed_quotes=config.get("allowed_quotes", []),
                min_daily_volume_quote=float(sf_cfg.get("min_volume_usd", 0) or 0),
                max_spread_pct=float(sf_cfg.get("max_spread_pct", 100) or 100),
                whitelist=discovered,
                blacklist=config.get("excluded_symbols"),
                max_pairs=config.get("top_n_symbols"),
            )
            config["tradable_symbols"] = tradable
            config["symbols"] = tradable + config.get("onchain_symbols", [])
            final_syms = config["symbols"]
            for tf in config.get("timeframes", []):
                logger.info(
                    "Final symbols for timeframe %s (%d): %s",
                    tf,
                    len(final_syms),
                    ", ".join(sorted(final_syms)),
                )
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
                config["tradable_symbols"] = list(cached)
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
                    config["tradable_symbols"] = list(fallback)
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
                    return MainResult(notifier, stop_reason)
        else:
            logger.error(
                "No symbols discovered after %d attempts; aborting startup",
                MAX_SYMBOL_SCAN_ATTEMPTS,
            )
            if status_updates:
                notifier.notify(
                    f"❌ Startup aborted after {MAX_SYMBOL_SCAN_ATTEMPTS} symbol scan attempts"
                )
            return MainResult(notifier, stop_reason)

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
        return MainResult(notifier, stop_reason)
    risk_config = build_risk_config(config, volume_ratio)
    risk_manager = RiskManager(risk_config)

    wallet: Wallet | None = None
    if config.get("execution_mode") == "dry_run":
        start_bal = float(
            os.getenv("PAPER_BALANCE") or config.get("paper_balance", 1000.0)
        )
        wallet = Wallet(
            start_bal,
            config.get("max_open_trades", 1),
            config.get("allow_short", False),
        )
        log_balance(wallet.total_balance)
        last_balance = notify_balance_change(
            notifier,
            last_balance,
            float(wallet.total_balance),
            balance_updates,
        )

    mode = user.get("mode", config.get("mode", "auto"))
    onchain_cfg = config.get("evaluation", {}).get("onchain_watchers", {})

    def start_onchain_watchers() -> None:
        register_task(asyncio.create_task(monitor_pump_raydium()))
        register_task(asyncio.create_task(periodic_mint_sanity_check()))

    if mode != "cex" and onchain_cfg.get("enabled", True):
        start_onchain_watchers()
    else:
        logger.info(
            "On-chain watchers disabled (mode=%s, enabled=%s)",
            mode,
            onchain_cfg.get("enabled", True),
        )

    max_open_trades = config.get("max_open_trades", 1)
    position_guard = OpenPositionGuard(max_open_trades)
    log_ml_status_once()
    rotator = PortfolioRotator()

    state = {"running": True, "mode": mode}
    # Caches for OHLCV and regime data are stored on the session_state
    session_state = SessionState(last_balance=last_balance)
    last_candle_ts: dict[str, int] = {}

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
    ctx.wallet = wallet
    # backwards compatibility for modules still expecting ``paper_wallet``
    ctx.paper_wallet = wallet
    ctx.position_guard = position_guard
    ctx.balance = await fetch_and_log_balance(exchange, wallet, config)

    # ------------------------------------------------------------------
    # OHLCV provider setup
    # ------------------------------------------------------------------
    async def _ohlcv_provider(
        symbol: str, timeframe: str, limit: int = 500
    ) -> pd.DataFrame:
        """Return OHLCV data for *symbol* and *timeframe*.

        The provider first attempts to read from the in-memory cache and
        falls back to a direct ``fetch_ohlcv`` call on the active exchange.
        The actual blocking ``fetch_ohlcv`` call is executed in a worker thread
        so the event loop remains responsive.
        """

        try:
            df = session_state.df_cache.get(timeframe, {}).get(symbol)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # limit rows if the cached DataFrame is larger than requested
                return df.tail(limit)
        except Exception:
            pass

        data = await asyncio.to_thread(
            exchange.fetch_ohlcv, symbol, timeframe, limit=limit
        )
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(data, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        update_df_cache(session_state.df_cache, timeframe, symbol, df)
        return df

    set_ohlcv_provider(_ohlcv_provider)

    async def status_loop() -> None:
        last_line = ""
        while True:
            try:
                balance = await fetch_balance(exchange, wallet, config)
            except Exception:
                balance = 0.0
            positions = getattr(wallet, "positions", {}) if wallet else {}
            line = format_monitor_line(ctx, session_state, balance, positions, lastlog.last)
            out_line = line[:180]
            if out_line != last_line:
                sys.stdout.write("\r" + out_line.ljust(180))
                sys.stdout.flush()
                last_line = out_line
            await asyncio.sleep(5)

    register_task(asyncio.create_task(status_loop()))

    if not config.get("hft"):
        register_task(asyncio.create_task(console_control.control_loop(state, ctx, session_state)))
    register_task(
        asyncio.create_task(
            _rotation_loop(
                rotator,
                exchange,
                user.get("wallet_address", ""),
                state,
                notifier,
                check_balance_change,
            )
        )
    )
    if config.get("solana_scanner", {}).get("enabled"):
        register_task(asyncio.create_task(solana_scan_loop()))
    register_task(
        asyncio.create_task(
            registry_update_loop(
                config.get("token_registry", {}).get("refresh_interval_minutes", 15)
            )
        )
    )
    print("Bot running. Type 'stop' to pause, 'start' to resume, 'quit' to exit.")

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
        register_task(telegram_bot.run_async())

    if config.get("meme_wave_sniper", {}).get("enabled"):
        register_task(start_runner(config.get("meme_wave_sniper", {})))
    sniper_cfg = config.get("meme_wave_sniper", {})
    if sniper_cfg.get("enabled"):
        register_task(asyncio.create_task(sniper_run(sniper_cfg)))

    if config.get("scan_in_background", True):
        session_state.scan_task = register_task(
            asyncio.create_task(
                initial_scan(
                    exchange,
                    config,
                    session_state,
                    notifier if status_updates else None,
                )
            )
        )
    else:
        await initial_scan(
            exchange,
            config,
            session_state,
            notifier if status_updates else None,
        )

    exit_when_idle = config.get("exit_when_idle", False)
    if exit_when_idle and not config.get("hft", False):
        return MainResult(notifier, stop_reason)

    if config.get("hft", False):
        selected_symbols = list(dict.fromkeys(config.get("symbols", [])))

        async def strategy_loop() -> None:
            try:
                loaded_strategies = load_strategies(
                    mode=config.get("mode", "cex")
                )
            except Exception:
                logger.exception("Strategy engine initialization failed")
                return
            if not loaded_strategies:
                logger.error("No strategies loaded; aborting HFT loop")
                return

            logger.info(
                "Loaded %d strategies for %d symbols; starting scoring loop...",
                len(loaded_strategies),
                len(selected_symbols),
            )

            scan_secs = int(config.get("scan_interval_seconds", 10))
            while not shutdown_event.is_set():
                try:
                    scores: dict[str, float] = {}
                    for strat in loaded_strategies:
                        try:
                            res = await strategies.score(
                                strat,
                                symbols=selected_symbols,
                                timeframes=config.get("timeframes", []),
                            )
                            for (sym, tf), val in res.items():
                                if isinstance(val, tuple):
                                    score, action = val
                                else:
                                    score, action = val, "none"
                                logger.info(
                                    "Signal for %s | %s | %s: %s, %s",
                                    strat.name,
                                    sym,
                                    tf,
                                    score,
                                    action,
                                )
                                scores[f"{strat.name}:{sym}:{tf}"] = score
                        except Exception:
                            logger.exception(
                                "Strategy %s scoring failed", getattr(strat, "name", str(strat))
                            )
                    logger.info(
                        "Top strategy scores:\n%s",
                        format_top(scores, n=25),
                    )
                except Exception:
                    logger.exception("Strategy scoring step failed")
                await asyncio.sleep(scan_secs)

        strategy_task = asyncio.create_task(strategy_loop(), name="strategy-loop")
        await strategy_task
        return MainResult(notifier, stop_reason)

    async def _eval_symbol(symbol: str, data: dict) -> dict:
        df_map = {
            tf: ctx.df_cache.get(tf, {}).get(symbol)
            for tf in data.get("timeframes", [])
        }
        res = await analyze_symbol(
            symbol,
            df_map,
            ctx.config.get("mode", "cex"),
            ctx.config,
            ctx.notifier,
            mempool_monitor=ctx.mempool_monitor,
            mempool_cfg=ctx.mempool_cfg,
        )
        if not res.get("skip"):
            ctx.analysis_results = [res]
            ctx.current_batch = [symbol]
            await execute_signals(ctx)
        return res

    async def _eval_wrapper(symbol: str, data: dict) -> dict:
        return await asyncio.wait_for(_eval_symbol(symbol, data), timeout=8)

    global stream_evaluator
    eval_cfg = SimpleNamespace(
        trading=SimpleNamespace(**config.get("trading", {})),
        evaluation=SimpleNamespace(**config.get("runtime", {}).get("evaluation", {})),
    )
    # Initialize the stream evaluator once
    stream_evaluator = StreamEvaluator(_eval_wrapper, cfg=eval_cfg)
    await stream_evaluator.start()
    set_stream_evaluator(stream_evaluator)

    runner = PhaseRunner(
        [
            fetch_candidates,
            update_caches,
            enrich_with_pyth,
            refresh_balance,
            handle_exits,
        ]
    )

    async def _ws_price_feed_listener() -> None:
        if not ws_client:
            return
        while True:
            try:
                msg = await ws_client._next_message(timeout=5.0)
                if msg is not None:
                    wake_event.set()
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1)

    async def _mempool_event_listener() -> None:
        if not mempool_monitor:
            return
        interval = float(mempool_cfg.get("poll_interval", 5))
        while True:
            try:
                await asyncio.wait_for(
                    mempool_monitor.fetch_priority_fee(),
                    timeout=interval,
                )
                wake_event.set()
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break
            except Exception:
                pass
            await asyncio.sleep(interval)

    listener_tasks: list[asyncio.Task] = []
    if ws_client:
        listener_tasks.append(asyncio.create_task(_ws_price_feed_listener()))
    if mempool_monitor:
        listener_tasks.append(asyncio.create_task(_mempool_event_listener()))

    async def run_evaluation_cycle() -> None:
        ctx.timing = await runner.run(ctx)

    try:
        logger.info("Continuous evaluation loop started")
        stop_reason_ref = [stop_reason]
        await evaluation_loop(run_evaluation_cycle, ctx, config, stop_reason_ref)
        stop_reason = stop_reason_ref[0]
    except asyncio.CancelledError:
        stop_reason = "external signal"
        raise
    finally:
        await stream_evaluator.drain()
        await stream_evaluator.stop()
        for task in listener_tasks:
            task.cancel()
            with contextlib.suppress(Exception):
                await task
        if hasattr(exchange, "close"):
            if asyncio.iscoroutinefunction(getattr(exchange, "close")):
                with contextlib.suppress(Exception):
                    await exchange.close()
            else:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(exchange.close)
        if ws_client and hasattr(ws_client, "close"):
            if asyncio.iscoroutinefunction(getattr(ws_client, "close")):
                with contextlib.suppress(Exception):
                    await ws_client.close()
            else:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(ws_client.close)
        if telegram_bot:
            telegram_bot.stop()
        for task in list(BACKGROUND_TASKS):
            task.cancel()
        await asyncio.gather(*BACKGROUND_TASKS, return_exceptions=True)
        BACKGROUND_TASKS.clear()
        WS_PING_TASKS.clear()
        SNIPER_TASKS.clear()
        CROSS_ARB_TASKS.clear()
        NEW_SOLANA_TOKENS.clear()

    if not state.get("running", True) and stop_reason == "completed":
        stop_reason = "state['running'] set to False"

    return MainResult(notifier, stop_reason)


def _reload_modules() -> None:
    """Reload project modules for hot-reload scenarios."""
    if not os.environ.get("CRYPTOBOT_HOT_RELOAD"):
        return
    import importlib

    for name, module in list(sys.modules.items()):
        if name.startswith("crypto_bot") or name.startswith("schema"):
            importlib.reload(module)


async def shutdown(calling_task: asyncio.Task | None = None) -> None:
    """Cancel and gather all running asyncio tasks except ``calling_task``."""
    caller = calling_task or asyncio.current_task()
    tasks = [t for t in asyncio.all_tasks() if t is not caller]
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def main() -> None:
    """Entry point for running the trading bot with error handling."""
    load_dotenv()
    _ensure_user_setup()
    load_dotenv(override=True)

    from crypto_bot.utils.ml_utils import init_ml_components

    global ScannerConfig, SolanaScannerConfig, PythConfig
    global TelegramNotifier, send_test_message, LOG_DIR, setup_logger
    global PortfolioRotator, load_or_create, analyze_symbol, dca_bot, cooldown_configure
    global BotContext, PhaseRunner, RiskManager, RiskConfig, calculate_trailing_stop, should_exit
    global cex_trade_async, get_exchange, get_exchanges, OpenPositionGuard
    global console_control, log_position, log_balance
    global load_kraken_symbols, update_ohlcv_cache, update_multi_tf_ohlcv_cache, update_regime_tf_cache
    global timeframe_seconds, market_loader_configure, fetch_order_book_async, WS_OHLCV_TIMEOUT
    global PAIR_FILE, load_liquid_pairs, DEFAULT_MIN_VOLUME_USD, DEFAULT_TOP_K, refresh_pairs_async
    global build_priority_queue, get_solana_new_tokens, get_filtered_symbols, fix_symbol, symbol_utils, symbol_cache_guard
    global log_cycle_metrics, PaperWallet, Wallet, compute_strategy_weights, optimize_strategies
    global write_cycle_metrics, TOKEN_MINTS, monitor_pump_raydium, refresh_mints, periodic_mint_sanity_check, fetch_from_helius
    global pnl_logger, regime_pnl_tracker, record_sol_scanner_metrics, registry
    global auto_convert_funds, check_wallet_balances, detect_non_trade_tokens
    global classify_regime_async, classify_regime_cached, calc_atr, monitor_price, SolanaMempoolMonitor, maybe_refresh_model
    global fetch_geckoterminal_ohlcv, fetch_solana_prices, cross_chain_trade, sniper_solana, sniper_trade
    global load_token_mints, set_token_mints, TelegramBotUI, start_runner, sniper_run
    global stream_evaluator

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
    from crypto_bot.execution.order_executor import (
        execute_trade_async as cex_trade_async,
    )
    from crypto_bot.execution.cex_executor import (
        get_exchange,
        get_exchanges,
    )
    from crypto_bot.open_position_guard import OpenPositionGuard
    from crypto_bot import console_control
    from crypto_bot.utils.position_logger import log_position, log_balance
    from crypto_bot.utils.market_loader import (
        load_kraken_symbols,
        update_ohlcv_cache,
        update_multi_tf_ohlcv_cache,
        update_regime_tf_cache,
        timeframe_seconds,
        configure as market_loader_configure,
        fetch_order_book_async,
        WS_OHLCV_TIMEOUT,
        fetch_geckoterminal_ohlcv,
    )
    from crypto_bot.utils.pair_cache import PAIR_FILE, load_liquid_pairs
    from tasks.refresh_pairs import (
        DEFAULT_MIN_VOLUME_USD,
        DEFAULT_TOP_K,
        refresh_pairs_async,
    )
    from crypto_bot.utils.eval_queue import build_priority_queue
    from crypto_bot.solana import (
        get_solana_new_tokens,
        fetch_solana_prices,
        sniper_solana,
        start_runner,
    )
    from crypto_bot.utils.symbol_utils import (
        get_filtered_symbols,
        fix_symbol,
        symbol_cache_guard,
    )
    from crypto_bot.utils import symbol_utils
    from crypto_bot.utils.metrics_logger import log_cycle as log_cycle_metrics
    from crypto_bot.paper_wallet import PaperWallet  # backward compatibility
    from wallet import Wallet
    from crypto_bot.utils.strategy_utils import compute_strategy_weights
    from crypto_bot.auto_optimizer import optimize_strategies
    from crypto_bot.utils.telemetry import write_cycle_metrics
    from crypto_bot.utils.token_registry import (
        TOKEN_MINTS,
        monitor_pump_raydium,
        refresh_mints,
        periodic_mint_sanity_check,
        fetch_from_helius,
        load_token_mints,
        set_token_mints,
    )
    from crypto_bot.utils import token_registry as registry
    from crypto_bot.utils import pnl_logger, regime_pnl_tracker
    from crypto_bot.solana_trading import cross_chain_trade, sniper_trade
    from crypto_bot.telegram_bot_ui import TelegramBotUI
    from crypto_bot.solana.runner import run as sniper_run
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
    from crypto_bot.regime.reloader import maybe_refresh_model
    from crypto_bot.volatility_filter import calc_atr
    from crypto_bot.solana.exit import monitor_price
    from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor

    init_ml_components()
    _reload_modules()

    notifier: TelegramNotifier | None = None
    reason = "completed"
    try:
        await refresh_mints()
        result = await _main_impl()
        notifier = result.notifier
        reason = result.reason
    except asyncio.CancelledError:
        reason = "external signal"
        raise
    except Exception as exc:  # pragma: no cover - error path
        reason = f"exception: {exc}"
        logger.exception("Unhandled error in main: %s", exc)
        if notifier:
            notifier.notify(f"❌ Bot stopped: {exc}")
    finally:
        logger.info("Bot shutting down")
        try:
            await shutdown(asyncio.current_task())
        except asyncio.CancelledError:
            logger.info("Shutdown cancelled")
            raise
        finally:
            if notifier:
                notifier.notify(f"Bot shutting down: {reason}")
            logger.info("Bot shutting down: %s", reason)
        msg = f"Bot shutting down: {reason}"
        logger.info(msg)
        if notifier:
            notifier.notify(msg)
        try:
            await shutdown()
        except Exception as exc:  # pragma: no cover - cleanup error
            logger.exception("Error during shutdown: %s", exc)
        finally:
            logger.info("Shutdown complete")


if __name__ == "__main__":  # pragma: no cover - manual execution
    asyncio.run(main())
