from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Dict

from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

from crypto_bot.portfolio_rotator import PortfolioRotator
from crypto_bot.utils.logger import setup_logger


class TelegramBotUI:
    """Simple Telegram UI for controlling the trading bot."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        state: Dict[str, object],
        log_file: Path | str,
        rotator: PortfolioRotator | None = None,
        exchange: object | None = None,
        wallet: str | None = None,
    ) -> None:
        self.token = token
        self.chat_id = chat_id
        self.state = state
        self.log_file = Path(log_file)
        self.rotator = rotator
        self.exchange = exchange
        self.wallet = wallet
        self.logger = setup_logger(__name__, "crypto_bot/logs/telegram_ui.log")

        self.updater = Updater(token=token, use_context=True)
        dp = self.updater.dispatcher
        dp.add_handler(CommandHandler("start", self.start_cmd))
        dp.add_handler(CommandHandler("stop", self.stop_cmd))
        dp.add_handler(CommandHandler("status", self.status_cmd))
        dp.add_handler(CommandHandler("log", self.log_cmd))
        dp.add_handler(CommandHandler("rotate_now", self.rotate_now_cmd))
        dp.add_handler(CommandHandler("toggle_mode", self.toggle_mode_cmd))

        self.thread: threading.Thread | None = None

    def run_async(self) -> None:
        """Start polling in a background thread."""
        self.thread = threading.Thread(target=self.updater.start_polling, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.updater.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    # Command handlers -------------------------------------------------
    def start_cmd(self, update: Update, context: CallbackContext) -> None:
        self.state["running"] = True
        update.message.reply_text("Trading started")

    def stop_cmd(self, update: Update, context: CallbackContext) -> None:
        self.state["running"] = False
        update.message.reply_text("Trading stopped")

    def status_cmd(self, update: Update, context: CallbackContext) -> None:
        running = self.state.get("running", False)
        mode = self.state.get("mode")
        update.message.reply_text(f"Running: {running}, mode: {mode}")

    def log_cmd(self, update: Update, context: CallbackContext) -> None:
        if self.log_file.exists():
            lines = self.log_file.read_text().splitlines()[-20:]
            text = "\n".join(lines) if lines else "(no logs)"
        else:
            text = "Log file not found"
        update.message.reply_text(text)

    def rotate_now_cmd(self, update: Update, context: CallbackContext) -> None:
        if not (self.rotator and self.exchange and self.wallet):
            update.message.reply_text("Rotation not configured")
            return
        try:
            bal = self.exchange.fetch_balance()
            holdings = {
                k: (v.get("total") if isinstance(v, dict) else v)
                for k, v in bal.items()
            }
            asyncio.run(
                self.rotator.rotate(
                    self.exchange,
                    self.wallet,
                    holdings,
                )
            )
            update.message.reply_text("Portfolio rotated")
        except Exception as exc:  # pragma: no cover - network
            self.logger.error("Rotation failed: %s", exc)
            update.message.reply_text("Rotation failed")

    def toggle_mode_cmd(self, update: Update, context: CallbackContext) -> None:
        mode = self.state.get("mode")
        mode = "onchain" if mode == "cex" else "cex"
        self.state["mode"] = mode
        update.message.reply_text(f"Mode set to {mode}")
