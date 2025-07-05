from __future__ import annotations

import asyncio
import threading
import time
import json
from pathlib import Path
from typing import Dict


import schedule
import json

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
)

from crypto_bot.portfolio_rotator import PortfolioRotator
from crypto_bot.utils.logger import setup_logger
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot import log_reader
from crypto_bot.utils.open_trades import get_open_trades

MENU = "MENU"
SIGNALS = "SIGNALS"
BALANCE = "BALANCE"
TRADES = "TRADES"

ASSET_SCORES_FILE = Path("crypto_bot/logs/asset_scores.json")
TRADES_FILE = Path("crypto_bot/logs/trades.csv")

SIGNALS_FILE = Path("crypto_bot/logs/asset_scores.json")
TRADES_FILE = Path("crypto_bot/logs/trades.csv")


class TelegramBotUI:
    """Simple Telegram UI for controlling the trading bot."""

    def __init__(
        self,
        notifier: TelegramNotifier,
        state: Dict[str, object],
        log_file: Path | str,
        rotator: PortfolioRotator | None = None,
        exchange: object | None = None,
        wallet: str | None = None,
    ) -> None:
        self.notifier = notifier
        self.token = notifier.token
        self.chat_id = notifier.chat_id
        self.state = state
        self.log_file = Path(log_file)
        self.rotator = rotator
        self.exchange = exchange
        self.wallet = wallet
        self.logger = setup_logger(__name__, "crypto_bot/logs/telegram_ui.log")

        self.app = ApplicationBuilder().token(self.token).build()
        self.app.add_handler(CommandHandler("start", self.start_cmd))
        self.app.add_handler(CommandHandler("stop", self.stop_cmd))
        self.app.add_handler(CommandHandler("status", self.status_cmd))
        self.app.add_handler(CommandHandler("log", self.log_cmd))
        self.app.add_handler(CommandHandler("rotate_now", self.rotate_now_cmd))
        self.app.add_handler(CommandHandler("toggle_mode", self.toggle_mode_cmd))
        self.app.add_handler(CommandHandler("menu", self.menu_cmd))
        self.app.add_handler(CommandHandler("signals", self.show_signals))
        self.app.add_handler(CommandHandler("balance", self.show_balance))
        self.app.add_handler(CommandHandler("trades", self.show_trades))
        self.app.add_handler(CallbackQueryHandler(self.menu_cmd, pattern=f"^{MENU}$"))
        self.app.add_handler(
            CallbackQueryHandler(self.show_signals, pattern=f"^{SIGNALS}$")
        )
        self.app.add_handler(
            CallbackQueryHandler(self.show_balance, pattern=f"^{BALANCE}$")
        )
        self.app.add_handler(
            CallbackQueryHandler(self.show_trades, pattern=f"^{TRADES}$")
        )

        self.scheduler_thread: threading.Thread | None = None

        schedule.every().day.at("00:00").do(self.send_daily_summary)
        self.scheduler_thread = threading.Thread(
            target=self._run_scheduler, daemon=True
        )
        self.scheduler_thread.start()

        self.task: asyncio.Task | None = None

    def run_async(self) -> None:
        """Start polling within the current event loop."""

        async def run() -> None:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()

        self.task = asyncio.create_task(run())

    def _run_scheduler(self) -> None:
        while True:
            schedule.run_pending()
            time.sleep(1)

    def stop(self) -> None:
        if self.task:
            self.task.cancel()
        schedule.clear()
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=2)

    # Command handlers -------------------------------------------------
    async def start_cmd(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        self.state["running"] = True
        await update.message.reply_text("Trading started")

    async def stop_cmd(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        self.state["running"] = False
        await update.message.reply_text("Trading stopped")

    async def status_cmd(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        running = self.state.get("running", False)
        mode = self.state.get("mode")
        await update.message.reply_text(f"Running: {running}, mode: {mode}")

    async def log_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if self.log_file.exists():
            lines = self.log_file.read_text().splitlines()[-20:]
            text = "\n".join(lines) if lines else "(no logs)"
        else:
            text = "Log file not found"
        await update.message.reply_text(text)

    async def rotate_now_cmd(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not (self.rotator and self.exchange and self.wallet):
            await update.message.reply_text("Rotation not configured")
            return
        try:
            bal = self.exchange.fetch_balance()
            holdings = {
                k: (v.get("total") if isinstance(v, dict) else v)
                for k, v in bal.items()
            }
            await self.rotator.rotate(
                self.exchange,
                self.wallet,
                holdings,
            )
            await update.message.reply_text("Portfolio rotated")
        except Exception as exc:  # pragma: no cover - network
            self.logger.error("Rotation failed: %s", exc)
            await update.message.reply_text("Rotation failed")

    def send_daily_summary(self) -> None:
        stats = log_reader.trade_summary("crypto_bot/logs/trades.csv")
        msg = (
            "Daily Summary\n"
            f"Trades: {stats['num_trades']}\n"
            f"Total PnL: {stats['total_pnl']:.2f}\n"
            f"Win rate: {stats['win_rate']*100:.1f}%\n"
            f"Active positions: {stats['active_positions']}"
        )
        err = self.notifier.notify(msg)
        if err:
            self.logger.error("Failed to send summary: %s", err)

    async def toggle_mode_cmd(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        mode = self.state.get("mode")
        mode = "onchain" if mode == "cex" else "cex"
        self.state["mode"] = mode
        await update.message.reply_text(f"Mode set to {mode}")

    async def menu_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        cmds = [
            "/start",
            "/stop",
            "/status",
            "/log",
            "/rotate_now",
            "/toggle_mode",
            "/signals",
            "/balance",
            "/trades",
        ]
        await update.message.reply_text("Available commands:\n" + "\n".join(cmds))

    async def show_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if SIGNALS_FILE.exists():
            try:
                data = json.loads(SIGNALS_FILE.read_text())
                lines = [f"{k}: {v:.2f}" for k, v in data.items()]
                text = "\n".join(lines) if lines else "(no signals)"
            except Exception:
                text = "Invalid signals file"
        else:
            text = "No signals found"
        await update.message.reply_text(text)

    async def show_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self.exchange:
            await update.message.reply_text("Exchange not configured")
            return
        try:
            bal = self.exchange.fetch_balance()
            lines = [
                f"{k}: {v.get('total') if isinstance(v, dict) else v}"
                for k, v in bal.items()
            ]
            text = "\n".join(lines) if lines else "(no balance)"
        except Exception as exc:  # pragma: no cover - network
            self.logger.error("Balance fetch failed: %s", exc)
            text = "Balance fetch failed"
        await update.message.reply_text(text)

    async def show_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if TRADES_FILE.exists():
            lines = TRADES_FILE.read_text().splitlines()[-20:]
            text = "\n".join(lines) if lines else "(no trades)"
        else:
            text = "No trades found"
        await update.message.reply_text(text)
    async def menu_cmd(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        markup = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Signals", callback_data=SIGNALS)],
                [InlineKeyboardButton("Wallet Balance", callback_data=BALANCE)],
                [InlineKeyboardButton("Open Trades", callback_data=TRADES)],
            ]
        )
        if update.message:
            await update.message.reply_text("Menu", reply_markup=markup)
        elif update.callback_query:
            await update.callback_query.answer()
            await update.callback_query.message.edit_text("Menu", reply_markup=markup)

    async def show_signals(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query:
            await query.answer()
        if ASSET_SCORES_FILE.exists():
            try:
                data = json.loads(ASSET_SCORES_FILE.read_text())
            except Exception:
                data = {}
        else:
            data = {}
        if data:
            lines = [f"{s} {v:+.4f}" for s, v in data.items()]
            text = "\n".join(lines)
        else:
            text = "No signals"
        markup = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Back", callback_data=MENU)]]
        )
        await query.message.edit_text(text, reply_markup=markup)

    async def show_balance(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query:
            await query.answer()
        text: str
        if not self.exchange:
            text = "Exchange not configured"
        else:
            try:
                bal = self.exchange.fetch_balance()
                usdt = bal.get("USDT")
                amount = float(
                    usdt.get("free", usdt) if isinstance(usdt, dict) else usdt or 0.0
                )
                text = f"Free USDT: {amount:.2f}"
            except Exception as exc:  # pragma: no cover - network
                self.logger.error("Balance fetch failed: %s", exc)
                text = "Failed to fetch balance"
        markup = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Back", callback_data=MENU)]]
        )
        await query.message.edit_text(text, reply_markup=markup)

    async def show_trades(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query:
            await query.answer()
        trades = get_open_trades(TRADES_FILE)
        if not trades:
            text = "No open trades"
        else:
            symbols = {t["symbol"] for t in trades}
            prices: dict[str, float] = {}
            for sym in symbols:
                try:
                    if asyncio.iscoroutinefunction(
                        getattr(self.exchange, "fetch_ticker", None)
                    ):
                        ticker = await self.exchange.fetch_ticker(sym)
                    else:
                        ticker = await asyncio.to_thread(
                            self.exchange.fetch_ticker, sym
                        )
                    price = ticker.get("last") or ticker.get("close") or 0.0
                    prices[sym] = float(price)
                except Exception:
                    prices[sym] = 0.0
            lines = []
            for t in trades:
                sym = t.get("symbol")
                entry = float(t.get("price", 0))
                amt = float(t.get("amount", 0))
                pnl = (prices.get(sym, 0.0) - entry) * amt
                lines.append(f"{sym} {pnl:+.2f}")
            text = "\n".join(lines)
        markup = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Back", callback_data=MENU)]]
        )
        await query.message.edit_text(text, reply_markup=markup)
