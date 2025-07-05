from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Dict


import schedule

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from crypto_bot.portfolio_rotator import PortfolioRotator
from crypto_bot.utils.logger import setup_logger
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot import log_reader


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
    async def start_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.state["running"] = True
        await update.message.reply_text("Trading started")

    async def stop_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.state["running"] = False
        await update.message.reply_text("Trading stopped")

    async def status_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

    async def rotate_now_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

    async def toggle_mode_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        mode = self.state.get("mode")
        mode = "onchain" if mode == "cex" else "cex"
        self.state["mode"] = mode
        await update.message.reply_text(f"Mode set to {mode}")
