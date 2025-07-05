import os
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml
from dotenv import dotenv_values

from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.utils.telegram import send_message, TelegramNotifier
from crypto_bot.utils.telegram import send_message, send_test_message
from crypto_bot.utils.trade_reporter import report_entry, report_exit
from crypto_bot.utils.logger import setup_logger
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
from crypto_bot import console_monitor
from crypto_bot.utils.performance_logger import log_performance
from crypto_bot.utils.strategy_utils import compute_strategy_weights
from crypto_bot.utils.position_logger import log_position, log_balance
from crypto_bot.utils.regime_logger import log_regime
from crypto_bot.utils.market_loader import (
    load_kraken_symbols,
    load_ohlcv_parallel,
    update_ohlcv_cache,
    fetch_ohlcv_async,
)
from crypto_bot.utils.symbol_pre_filter import filter_symbols
from crypto_bot.utils.symbol_utils import get_filtered_symbols
from crypto_bot.utils.pnl_logger import log_pnl
from crypto_bot.utils.strategy_analytics import write_scores
from crypto_bot.utils.regime_pnl_tracker import log_trade as log_regime_pnl
from crypto_bot.utils.trend_confirmation import confirm_multi_tf_trend
from crypto_bot.regime.regime_classifier import CONFIG


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
ENV_PATH = Path(__file__).resolve().parent / ".env"

logger = setup_logger("bot", "crypto_bot/logs/bot.log", to_console=False)


def direction_to_side(direction: str) -> str:
    """Translate strategy direction to trade side."""
    return "buy" if direction == "long" else "sell"


def opposite_side(side: str) -> str:
    """Return the opposite trading side."""
    return "sell" if side == "buy" else "buy"


def load_config() -> dict:
    """Load YAML configuration for the bot."""
    with open(CONFIG_PATH) as f:
        logger.info("Loading config from %s", CONFIG_PATH)
        return yaml.safe_load(f)


async def main() -> None:
    """Entry point for running the trading bot."""

    logger.info("Starting bot")
    config = load_config()
    notifier = TelegramNotifier.from_config(config.get("telegram", {}))
    notifier.notify("🤖 CoinTrader2.0 started")
    volume_ratio = 0.01 if config.get("testing_mode") else 1.0
    cooldown_configure(config.get("min_cooldown", 0))
    secrets = dotenv_values(ENV_PATH)
    os.environ.update(secrets)

    user = load_or_create()
    notifier: TelegramNotifier | None = None
    if user.get("telegram_token") and user.get("telegram_chat_id"):
        notifier = TelegramNotifier(user["telegram_token"], user["telegram_chat_id"])

    if user.get("telegram_token") and user.get("telegram_chat_id"):
        if not send_test_message(
            user["telegram_token"],
            user["telegram_chat_id"],
            "Bot started",
        ):
            logger.warning(
                "Telegram test message failed; check your token and chat ID"
            )

    # allow user-configured exchange to override YAML setting
    if user.get("exchange"):
        config["exchange"] = user["exchange"]

    exchange, ws_client = get_exchange(config)

    if config.get("scan_markets", False) and not config.get("symbols"):
        config["symbols"] = await load_kraken_symbols(
            exchange, config.get("excluded_symbols", [])
        )

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
    except Exception as exc:  # pragma: no cover - network
        logger.error("Exchange API setup failed: %s", exc)
        err = notifier.notify(f"API error: {exc}")
        if err:
            logger.error("Failed to notify user: %s", err)
        return
    risk_params = {**config.get("risk", {})}
    risk_params.update(config.get("sentiment_filter", {}))
    risk_params.update(config.get("volatility_filter", {}))
    risk_params["symbol"] = config.get("symbol", "")
    risk_params["trade_size_pct"] = config.get("trade_size_pct", 0.1)
    risk_params["strategy_allocation"] = config.get("strategy_allocation", {})
    risk_params["volume_threshold_ratio"] = config.get("risk", {}).get(
        "volume_threshold_ratio", 0.1
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
        paper_wallet = PaperWallet(start_bal)
        log_balance(paper_wallet.balance)

    monitor_task = asyncio.create_task(
        console_monitor.monitor_loop(
            exchange, paper_wallet, "crypto_bot/logs/bot.log"
        )
    )

    open_side = None
    entry_price = None
    entry_time = None
    entry_regime = None
    entry_strategy = None
    entry_confidence = 0.0
    realized_pnl = 0.0
    trailing_stop = 0.0
    position_size = 0.0
    highest_price = 0.0
    current_strategy = None
    active_strategy = None
    stats_file = Path("crypto_bot/logs/strategy_stats.json")
    # File tracking individual trade performance used for analytics
    perf_file = Path("crypto_bot/logs/strategy_performance.json")
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}
    scores_file = Path("crypto_bot/logs/strategy_scores.json")

    rotator = PortfolioRotator()
    last_rotation = 0.0
    last_optimize = 0.0
    last_weight_update = 0.0

    mode = user.get("mode", config.get("mode", "auto"))
    state = {"running": True, "mode": mode}
    df_cache: dict[str, pd.DataFrame] = {}

    telegram_bot = None
    if notifier.enabled:
    notifier = None
    if user.get("telegram_token") and user.get("telegram_chat_id"):
        from crypto_bot.telegram_bot_ui import TelegramBotUI

        telegram_bot = TelegramBotUI(
            notifier,
            state,
            "crypto_bot/logs/bot.log",
            rotator,
            exchange,
            user.get("wallet_address", ""),
        )
        telegram_bot.run_async()
        notifier = TelegramNotifier(user["telegram_token"], user["telegram_chat_id"])

    while True:
        mode = state["mode"]

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

        if rotator.config.get("enabled"):
            if (
                time.time() - last_rotation
                >= rotator.config.get("interval_days", 7) * 86400
            ):
                bal = await asyncio.to_thread(exchange.fetch_balance)
                holdings = {
                    k: (v.get("total") if isinstance(v, dict) else v)
                    for k, v in bal.items()
                }
                await rotator.rotate(
                    exchange,
                    user.get("wallet_address", ""),
                    holdings,
                    notifier,
                )
                last_rotation = time.time()

        best = None
        best_score = -1.0
        df_current = None

        symbols = await get_filtered_symbols(exchange, config)
        df_cache = await update_ohlcv_cache(
            exchange,
            df_cache,
            symbols,
            timeframe=config["timeframe"],
            limit=100,
            use_websocket=config.get("use_websocket", False),
            force_websocket_history=config.get("force_websocket_history", False),
            max_concurrent=config.get("max_concurrent_ohlcv"),
        )

        tasks = []
        for sym in symbols:
            logger.info("🔹 Symbol: %s", sym)
            total_pairs += 1
            df_sym = df_cache.get(sym)
            if df_sym is None or df_sym.empty:
                logger.error("OHLCV fetch failed for %s", sym)
                continue

            # ensure we have a proper DataFrame with the expected columns
            expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not isinstance(df_sym, pd.DataFrame):
                df_sym = pd.DataFrame(df_sym, columns=expected_cols)
            elif not set(expected_cols).issubset(df_sym.columns):
                df_sym = pd.DataFrame(df_sym.to_numpy(), columns=expected_cols)
            logger.info("Fetched %d candles for %s", len(df_sym), sym)
            if sym == config.get("symbol"):
                df_current = df_sym
            tasks.append(analyze_symbol(sym, df_sym, mode, config, notifier))

        results = await asyncio.gather(*tasks)

        scalpers = [
            r["symbol"]
            for r in results
            if r.get("name") in {"micro_scalp", "bounce_scalper"}
        ]
        if scalpers:
            df_cache = await update_ohlcv_cache(
                exchange,
                df_cache,
                scalpers,
                timeframe=config.get("scalp_timeframe", "1m"),
                limit=100,
                use_websocket=config.get("use_websocket", False),
                force_websocket_history=config.get("force_websocket_history", False),
                max_concurrent=config.get("max_concurrent_ohlcv"),
            )
            tasks = [
                analyze_symbol(sym, df_cache.get(sym), mode, config, notifier)
                for sym in scalpers
            ]
            scalper_results = await asyncio.gather(*tasks)
            mapping = {r["symbol"]: r for r in scalper_results}
            results = [mapping.get(r["symbol"], r) for r in results]
            if config.get("symbol") in mapping:
                df_current = df_cache.get(config["symbol"])

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

            if open_side is None and in_cooldown(sym, name_sym):
                continue

            params_file = Path("crypto_bot/logs/optimized_params.json")
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

            if direction_sym != "none" and score_sym > best_score:
                best_score = score_sym
                best = {
                    "symbol": sym,
                    "df": df_sym,
                    "regime": regime_sym,
                    "env": env_sym,
                    "name": name_sym,
                    "direction": direction_sym,
                    "score": score_sym,
                }

        if open_side and df_current is None:
            # ensure current market data is loaded
            try:
                if config.get("use_websocket", False) and hasattr(exchange, "watch_ohlcv"):
                    data = await exchange.watch_ohlcv(
                        config["symbol"], timeframe=config["timeframe"], limit=100
                    )
                else:
                    if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
                        data = await exchange.fetch_ohlcv(
                            config["symbol"], timeframe=config["timeframe"], limit=100
                        )
                    else:
                        data = await asyncio.to_thread(
                            exchange.fetch_ohlcv,
                            config["symbol"],
                            timeframe=config["timeframe"],
                            limit=100,
                        )
            except Exception as exc:  # pragma: no cover - network
                logger.error("OHLCV fetch failed for %s: %s", config["symbol"], exc)
                continue

            if data and len(data[0]) > 6:
                data = [
                    [c[0], c[1], c[2], c[3], c[4], c[6]] for c in data
                ]
            df_current = pd.DataFrame(
                data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

        if open_side and df_current is not None:
            current_price = df_current["close"].iloc[-1]
        elif best:
            current_price = best["df"]["close"].iloc[-1]
        else:
            current_price = None

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

        if open_side:
            pnl_pct = ((current_price - entry_price) / entry_price) * (
                1 if open_side == "buy" else -1
            )
            if paper_wallet:
                unreal = paper_wallet.unrealized(current_price)
                logger.info(
                    "Paper balance %.2f USDT (Unrealized %.2f)",
                    paper_wallet.balance + unreal,
                    unreal,
                )
            if pnl_pct >= config["exit_strategy"]["min_gain_to_trail"]:
                if current_price > highest_price:
                    highest_price = current_price
                if highest_price:
                    trailing_stop = calculate_trailing_stop(
                        pd.Series([highest_price]),
                        config["exit_strategy"]["trailing_stop_pct"],
                    )

            df_to_use = df_current if df_current is not None else best["df"]
            exit_signal, trailing_stop = should_exit(
                df_to_use,
                current_price,
                trailing_stop,
                config,
                risk_manager,
            )
            if exit_signal:
                pct = get_partial_exit_percent(pnl_pct * 100)
                sell_amount = (
                    position_size * (pct / 100)
                    if config["exit_strategy"]["scale_out"] and pct > 0
                    else position_size
                )
                logger.info("Executing exit trade amount %.4f", sell_amount)
                await cex_trade_async(
                    exchange,
                    ws_client,
                    config["symbol"],
                    opposite_side(open_side),
                    sell_amount,
                    notifier,
                    dry_run=config["execution_mode"] == "dry_run",
                    use_websocket=config.get("use_websocket", False),
                    config=config,
                )
                if paper_wallet:
                    paper_wallet.close(sell_amount, current_price)
                realized_pnl += (
                    (current_price - entry_price)
                    * sell_amount
                    * (1 if open_side == "buy" else -1)
                )
                if sell_amount >= position_size:
                    risk_manager.cancel_stop_order(exchange)
                    risk_manager.deallocate_capital(
                        current_strategy, sell_amount * entry_price
                    )
                    log_performance(
                        {
                            "symbol": config["symbol"],
                            "regime": entry_regime,
                            "strategy": entry_strategy,
                            "pnl": realized_pnl,
                            "entry_time": entry_time,
                            "exit_time": datetime.utcnow().isoformat(),
                        }
                    )
                    log_pnl(
                        entry_strategy or "",
                        config["symbol"],
                        entry_price or 0.0,
                        current_price,
                        realized_pnl,
                        entry_confidence,
                        open_side or "",
                    )
                    log_regime_pnl(
                        entry_regime or "unknown",
                        entry_strategy or "",
                        realized_pnl,
                    )
                    if paper_wallet:
                        logger.info(
                            "Paper balance closed: %.2f USDT", paper_wallet.balance
                        )
                    if config["execution_mode"] != "dry_run":
                        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
                            bal = await exchange.fetch_balance()
                        else:
                            bal = await asyncio.to_thread(exchange.fetch_balance)
                        latest_balance = bal["USDT"]["free"]
                    else:
                        latest_balance = paper_wallet.balance if paper_wallet else 0.0
                    log_balance(float(latest_balance))
                    if notifier.enabled:
                    if notifier:
                        report_exit(
                            notifier,
                            config.get("symbol", ""),
                            entry_strategy or "",
                            realized_pnl,
                            "long" if open_side == "buy" else "short",
                        )
                    open_side = None
                    entry_price = None
                    entry_time = None
                    entry_regime = None
                    entry_strategy = None
                    entry_confidence = 0.0
                    realized_pnl = 0.0
                    position_size = 0.0
                    trailing_stop = 0.0
                    highest_price = 0.0
                    current_strategy = None
                    mark_cooldown(config["symbol"], active_strategy or name)
                    active_strategy = None
                else:
                    position_size -= sell_amount
                    risk_manager.deallocate_capital(
                        current_strategy, sell_amount * entry_price
                    )
                    risk_manager.update_stop_order(position_size)

        if score < config["signal_threshold"] or direction == "none":
            sym_to_log = best["symbol"] if best else config["symbol"]
            logger.info(
                "Skipping trade for %s \u2013 score %.2f below threshold %.2f or direction none",
                sym_to_log,
                score,
                config["signal_threshold"],
            )
            logger.info("No trade executed")
            rejected_score += 1
            logger.info(
                "Loop Summary: %s evaluated | %s trades | %s volume fails | %s score fails | %s unknown regime",
                total_pairs,
                trades_executed,
                rejected_volume,
                rejected_score,
                rejected_regime,
            )
            if direction == "none":
                regime_rejections += 1
            else:
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
            logger.info("Sleeping for %s minutes", config["loop_interval_minutes"])
            await asyncio.sleep(config["loop_interval_minutes"] * 60)
            continue

        if config["execution_mode"] != "dry_run":
            if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
                bal = await exchange.fetch_balance()
            else:
                bal = await asyncio.to_thread(exchange.fetch_balance)
            balance = bal["USDT"]["free"]
        else:
            balance = paper_wallet.balance if paper_wallet else 0.0

        if open_side and current_price is not None:
            pnl = (current_price - entry_price) * position_size
            if open_side == "sell":
                pnl = -pnl
            if paper_wallet:
                log_bal = paper_wallet.balance + pnl
            else:
                log_bal = balance + pnl
            log_position(
                config.get("symbol", ""),
                open_side,
                position_size,
                entry_price,
                current_price,
                log_bal,
            )
        if best:
            risk_manager.config.symbol = best["symbol"]
        df_for_size = best["df"] if best else None
        size = risk_manager.position_size(score, balance, df_for_size)
        if current_price and current_price > 0:
            order_amount = size / current_price
        else:
            order_amount = 0.0
        if open_side and trade_side == open_side:
            higher_tf = CONFIG.get("higher_timeframe", "4h")
            try:
                data_high = await fetch_ohlcv_async(
                    exchange,
                    config["symbol"],
                    timeframe=higher_tf,
                    limit=100,
                    use_websocket=config.get("use_websocket", False),
                    force_websocket_history=config.get("force_websocket_history", False),
                )
            except Exception:
                data_high = []
            if isinstance(data_high, Exception) or not data_high:
                logger.info("Skipping scale-in due to missing higher timeframe data")
                await asyncio.sleep(config["loop_interval_minutes"] * 60)
                continue
            if len(data_high[0]) > 6:
                data_high = [[c[0], c[1], c[2], c[3], c[4], c[6]] for c in data_high]
            df_high = pd.DataFrame(
                data_high,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            low_df = df_for_size if df_for_size is not None else df_current
            if not confirm_multi_tf_trend(low_df, df_high):
                logger.info("Scale-in blocked: trend not confirmed across timeframes")
                await asyncio.sleep(config["loop_interval_minutes"] * 60)
                continue
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
            logger.info("Sleeping for %s minutes", config["loop_interval_minutes"])
            await asyncio.sleep(config["loop_interval_minutes"] * 60)
            continue

        if env == "onchain":
            logger.info(
                "Executing %s entry %s %.4f at %.4f",
                name,
                direction,
                order_amount,
                current_price,
            )
            swap_result = await execute_swap(
                "SOL",
                "USDC",
                order_amount,
                notifier,
                slippage_bps=config.get("solana_slippage_bps", 50),
                dry_run=config["execution_mode"] == "dry_run",
            )
            if swap_result:
                logger.info(
                    "On-chain swap tx=%s in=%s out=%s amount=%s dry_run=%s",
                    swap_result.get("tx_hash"),
                    swap_result.get("token_in"),
                    swap_result.get("token_out"),
                    swap_result.get("amount"),
                    config["execution_mode"] == "dry_run",
                )
            risk_manager.register_stop_order(
                {
                    "token_in": "SOL",
                    "token_out": "USDC",
                    "amount": order_amount,
                    "dry_run": config["execution_mode"] == "dry_run",
                },
                strategy=strategy_name(regime, env),
                symbol="SOL/USDC",
                entry_price=current_price,
                confidence=score,
                direction=trade_side,
            )
            if paper_wallet:
                paper_wallet.open(trade_side, order_amount, current_price)
        else:
            logger.info(
                "Executing %s entry %s %.4f at %.4f",
                name,
                direction,
                order_amount,
                current_price,
            )
            config["symbol"] = best["symbol"] if best else config["symbol"]
            order = await cex_trade_async(
                exchange,
                ws_client,
                config["symbol"],
                trade_side,
                order_amount,
                notifier,
                dry_run=config["execution_mode"] == "dry_run",
                use_websocket=config.get("use_websocket", False),
                config=config,
            )
            if order:
                logger.info(
                    "CEX trade result - id=%s side=%s amount=%s price=%s dry_run=%s",
                    order.get("id"),
                    order.get("side"),
                    order.get("amount"),
                    order.get("price") or order.get("average"),
                    order.get("dry_run", config["execution_mode"] == "dry_run"),
                )
            stop_price = current_price * (
                1 - risk_manager.config.stop_loss_pct
                if trade_side == "buy"
                else 1 + risk_manager.config.stop_loss_pct
            )
            stop_order = place_stop_order(
                exchange,
                config["symbol"],
                "sell" if trade_side == "buy" else "buy",
                order_amount,
                stop_price,
                notifier,
                dry_run=config["execution_mode"] == "dry_run",
            )
            risk_manager.register_stop_order(
                stop_order,
                strategy=strategy_name(regime, env),
                symbol=config["symbol"],
                entry_price=current_price,
                confidence=score,
                direction=trade_side,
            )
            risk_manager.allocate_capital(name, size)
            if paper_wallet:
                paper_wallet.open(trade_side, order_amount, current_price)
            open_side = trade_side
            entry_price = current_price
            position_size = order_amount
            realized_pnl = 0.0
            entry_time = datetime.utcnow().isoformat()
            entry_regime = regime
            entry_strategy = strategy_name(regime, env)
            entry_confidence = score
            highest_price = entry_price
            current_strategy = name
            active_strategy = name
            log_bal = (
                paper_wallet.balance if config["execution_mode"] == "dry_run" else balance
            )
            log_position(
                config.get("symbol", ""),
                open_side,
                position_size,
                entry_price,
                current_price,
                log_bal,
            )
            if notifier.enabled:
            if notifier:
                report_entry(
                    notifier,
                    config.get("symbol", ""),
                    strategy_name(regime, env),
                    score,
                    direction,
                )
            logger.info("Trade opened at %.4f", entry_price)
            trades_executed += 1

        key = f"{env}_{regime}"
        stats.setdefault(key, {"trades": 0})
        stats[key]["trades"] += 1
        stats_file.write_text(json.dumps(stats))
        write_scores(scores_file, perf_file)
        logger.info("Updated trade stats %s", stats[key])

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
        logger.info("Sleeping for %s minutes", config["loop_interval_minutes"])
        await asyncio.sleep(config["loop_interval_minutes"] * 60)

    monitor_task.cancel()
    if telegram_bot:
        telegram_bot.stop()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
    if hasattr(exchange, "close"):
        if asyncio.iscoroutinefunction(getattr(exchange, "close")):
            await exchange.close()
        else:
            await asyncio.to_thread(exchange.close)


if __name__ == "__main__":  # pragma: no cover - manual execution
    asyncio.run(main())
