import os
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml
from dotenv import dotenv_values

from crypto_bot.utils.telegram import send_message
from crypto_bot.utils.logger import setup_logger
from crypto_bot.portfolio_rotator import PortfolioRotator
from crypto_bot.auto_optimizer import optimize_strategies
from crypto_bot.wallet_manager import load_or_create
from crypto_bot.regime.regime_classifier import classify_regime
from crypto_bot.strategy_router import route, strategy_name
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
from crypto_bot.utils.performance_logger import log_performance
from crypto_bot.utils.market_loader import load_kraken_symbols


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
ENV_PATH = Path(__file__).resolve().parent / ".env"

logger = setup_logger("bot", "crypto_bot/logs/bot.log")


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
    volume_ratio = 0.01 if config.get("testing_mode") else 1.0
    cooldown_configure(config.get("min_cooldown", 0))
    secrets = dotenv_values(ENV_PATH)
    os.environ.update(secrets)

    user = load_or_create()

    # allow user-configured exchange to override YAML setting
    if user.get("exchange"):
        config["exchange"] = user["exchange"]

    exchange, ws_client = get_exchange(config)

    if config.get("scan_markets", False) and not config.get("symbols"):
        config["symbols"] = load_kraken_symbols(
            exchange, config.get("excluded_symbols", [])
        )

    try:
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
            await exchange.fetch_balance()
        else:
            await asyncio.to_thread(exchange.fetch_balance)
    except Exception as exc:  # pragma: no cover - network
        logger.error("Exchange API setup failed: %s", exc)
        send_message(
            secrets.get("TELEGRAM_TOKEN"),
            config.get("telegram", {}).get("chat_id", ""),
            f"API error: {exc}",
        )
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

    open_side = None
    entry_price = None
    entry_time = None
    entry_regime = None
    entry_strategy = None
    realized_pnl = 0.0
    trailing_stop = 0.0
    position_size = 0.0
    highest_price = 0.0
    current_strategy = None
    active_strategy = None
    stats_file = Path("crypto_bot/logs/strategy_stats.json")
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}

    rotator = PortfolioRotator()
    last_rotation = 0.0
    last_optimize = 0.0

    mode = user.get("mode", config.get("mode", "auto"))
    state = {"running": True, "mode": mode}

    telegram_bot = None
    if user.get("telegram_token") and config.get("telegram", {}).get("chat_id"):
        from crypto_bot.telegram_bot_ui import TelegramBotUI

        telegram_bot = TelegramBotUI(
            user["telegram_token"],
            config["telegram"]["chat_id"],
            state,
            "crypto_bot/logs/bot.log",
            rotator,
            exchange,
            user.get("wallet_address", ""),
        )
        telegram_bot.run_async()

    while True:
        mode = state["mode"]

        total_pairs = 0
        signals_generated = 0
        trades_executed = 0
        trades_skipped = 0

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
                telegram_token=user.get("telegram_token", ""),
                chat_id=config.get("telegram", {}).get("chat_id", ""),
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
                    user.get("telegram_token", ""),
                    config.get("telegram", {}).get("chat_id", ""),
                )
                last_rotation = time.time()

        best = None
        best_score = -1.0
        df_current = None

        for sym in config.get("symbols", [config.get("symbol")]):
            logger.info("🔹 Symbol: %s", sym)
            total_pairs += 1
            try:
                if config.get("use_websocket", False) and hasattr(exchange, "watch_ohlcv"):
                    data = await exchange.watch_ohlcv(
                        sym, timeframe=config["timeframe"], limit=100
                    )
                    if data:
                        logger.debug("WS candle raw: %s", data[-1])
                else:
                    if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ohlcv", None)):
                        data = await exchange.fetch_ohlcv(
                            sym, timeframe=config["timeframe"], limit=100
                        )
                    else:
                        data = await asyncio.to_thread(
                            exchange.fetch_ohlcv,
                            sym,
                            timeframe=config["timeframe"],
                            limit=100,
                        )
            except Exception as exc:  # pragma: no cover - network
                logger.error("OHLCV fetch failed for %s: %s", sym, exc)
                continue

            if data and len(data[0]) > 6:
                data = [
                    [c[0], c[1], c[2], c[3], c[4], c[6]] for c in data
                ]
            df_sym = pd.DataFrame(
                data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            logger.info("Fetched %d candles for %s", len(df_sym), sym)

            if sym == config.get("symbol"):
                df_current = df_sym

            risk_manager.config.symbol = sym

            regime_sym = classify_regime(df_sym)
            logger.info("Market regime for %s classified as %s", sym, regime_sym)
            env_sym = mode if mode != "auto" else "cex"
            strategy_fn = route(regime_sym, env_sym)
            name_sym = strategy_name(regime_sym, env_sym)
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

            score_sym, direction_sym = await evaluate_async(strategy_fn, df_sym, config)
            logger.info("Signal %s %.2f %s", sym, score_sym, direction_sym)
            if direction_sym != "none":
                signals_generated += 1

            allowed, reason = risk_manager.allow_trade(df_sym)
            if not allowed:
                logger.info(
                    "Trade not allowed for %s \u2013 %s", sym, reason
                )
                if "Volume" in reason:
                    trades_skipped += 1
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

            exit_signal, trailing_stop = should_exit(
                df_current or best["df"], current_price, trailing_stop, config
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
                    secrets["TELEGRAM_TOKEN"],
                    config["telegram"]["chat_id"],
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
                    risk_manager.deallocate_capital(current_strategy, sell_amount)
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
                    if paper_wallet:
                        logger.info(
                            "Paper balance closed: %.2f USDT", paper_wallet.balance
                        )
                    open_side = None
                    entry_price = None
                    entry_time = None
                    entry_regime = None
                    entry_strategy = None
                    realized_pnl = 0.0
                    position_size = 0.0
                    trailing_stop = 0.0
                    highest_price = 0.0
                    current_strategy = None
                    mark_cooldown(config["symbol"], active_strategy or name)
                    active_strategy = None
                else:
                    position_size -= sell_amount
                    risk_manager.deallocate_capital(current_strategy, sell_amount)
                    risk_manager.update_stop_order(position_size)

        if score < config["signal_threshold"] or direction == "none":
            sym_to_log = best["symbol"] if best else config["symbol"]
            logger.info(
                "Skipping trade for %s \u2013 score %.2f below threshold %.2f or direction none",
                sym_to_log,
                score,
                config["signal_threshold"],
            )
            logger.info(
                f"Cycle Summary: {total_pairs} pairs evaluated, {signals_generated} signals, {trades_executed} trades executed, {trades_skipped} skipped due to volume."
            )
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
        if best:
            risk_manager.config.symbol = best["symbol"]
        size = risk_manager.position_size(score, balance)
        if not risk_manager.can_allocate(name, size, balance):
            logger.info("Capital cap reached for %s, skipping", name)
            logger.info(
                f"Cycle Summary: {total_pairs} pairs evaluated, {signals_generated} signals, {trades_executed} trades executed, {trades_skipped} skipped due to volume."
            )
            await asyncio.sleep(config["loop_interval_minutes"] * 60)
            continue

        if env == "onchain":
            logger.info(
                "Executing %s entry %s %.4f at %.4f",
                name,
                direction,
                size,
                current_price,
            )
            await execute_swap(
                "SOL",
                "USDC",
                size,
                user["telegram_token"],
                user["telegram_chat_id"],
                slippage_bps=config.get("solana_slippage_bps", 50),
                dry_run=config["execution_mode"] == "dry_run",
            )
            risk_manager.register_stop_order(
                {
                    "token_in": "SOL",
                    "token_out": "USDC",
                    "amount": size,
                    "dry_run": config["execution_mode"] == "dry_run",
                }
            )
            if paper_wallet:
                paper_wallet.open(trade_side, size, current_price)
        else:
            logger.info(
                "Executing %s entry %s %.4f at %.4f",
                name,
                direction,
                size,
                current_price,
            )
            config["symbol"] = best["symbol"] if best else config["symbol"]
            await cex_trade_async(
                exchange,
                ws_client,
                config["symbol"],
                trade_side,
                size,
                user["telegram_token"],
                user["telegram_chat_id"],
                dry_run=config["execution_mode"] == "dry_run",
                use_websocket=config.get("use_websocket", False),
                config=config,
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
                size,
                stop_price,
                user["telegram_token"],
                user["telegram_chat_id"],
                dry_run=config["execution_mode"] == "dry_run",
            )
            risk_manager.register_stop_order(stop_order)
            risk_manager.allocate_capital(name, size)
            if paper_wallet:
                paper_wallet.open(trade_side, size, current_price)
            open_side = trade_side
            entry_price = current_price
            position_size = size
            realized_pnl = 0.0
            entry_time = datetime.utcnow().isoformat()
            entry_regime = regime
            entry_strategy = strategy_name(regime, env)
            highest_price = entry_price
            current_strategy = name
            active_strategy = name
            logger.info("Trade opened at %.4f", entry_price)
            trades_executed += 1

        key = f"{env}_{regime}"
        stats.setdefault(key, {"trades": 0})
        stats[key]["trades"] += 1
        stats_file.write_text(json.dumps(stats))
        logger.info("Updated trade stats %s", stats[key])

        logger.info(
            f"Cycle Summary: {total_pairs} pairs evaluated, {signals_generated} signals, {trades_executed} trades executed, {trades_skipped} skipped due to volume."
        )
        await asyncio.sleep(config["loop_interval_minutes"] * 60)


if __name__ == "__main__":  # pragma: no cover - manual execution
    asyncio.run(main())
