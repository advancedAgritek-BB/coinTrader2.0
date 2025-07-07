from __future__ import annotations

"""Telegram command handlers used by TelegramBotUI and other clients."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from . import console_monitor, log_reader
from .utils.open_trades import get_open_trades

STRATEGY_FILE = Path("crypto_bot/logs/strategy_stats.json")
TRADES_FILE = Path("crypto_bot/logs/trades.csv")
LOG_FILE = Path("crypto_bot/logs/bot.log")
CONFIG_FILE = Path("crypto_bot/config.yaml")


def is_admin(update: Update, admin_id: str) -> bool:
    """Return True if the update came from the configured admin chat."""
    user_id = str(getattr(update.effective_user, "id", ""))
    return user_id == str(admin_id)


class BotController:
    """High level bot control and status retrieval."""

    def __init__(
        self,
        state: Dict[str, Any],
        exchange: Any = None,
        *,
        log_file: Path = LOG_FILE,
        trades_file: Path = TRADES_FILE,
        strategy_file: Path = STRATEGY_FILE,
        config_file: Path = CONFIG_FILE,
    ) -> None:
        self.state = state
        self.exchange = exchange
        self.log_file = Path(log_file)
        self.trades_file = Path(trades_file)
        self.strategy_file = Path(strategy_file)
        self.config_file = Path(config_file)

    async def start(self) -> str:
        self.state["running"] = True
        return "Trading started"

    async def stop(self) -> str:
        self.state["running"] = False
        return "Trading stopped"

    async def status(self) -> str:
        running = self.state.get("running", False)
        mode = self.state.get("mode")
        return f"Running: {running}, mode: {mode}"

    async def strategies(self) -> str:
        if self.strategy_file.exists():
            try:
                data = json.loads(self.strategy_file.read_text())
                lines = [f"{k}: {v}" for k, v in data.items()]
                return "\n".join(lines) if lines else "(no strategies)"
            except Exception:
                return "Invalid strategy file"
        return "No strategies found"

    async def positions(self) -> str:
        lines = await console_monitor.trade_stats_lines(self.exchange, self.trades_file)
        return "\n".join(lines) if lines else "(no positions)"

    async def logs(self) -> str:
        if self.log_file.exists():
            lines = self.log_file.read_text().splitlines()[-20:]
            return "\n".join(lines) if lines else "(no logs)"
        return "Log file not found"

    async def settings(self) -> str:
        if self.config_file.exists():
            return self.config_file.read_text()
        return "Config not found"


def _reply_or_edit(update: Update, text: str, reply_markup: Any | None = None) -> None:
    """Reply to message or edit callback text."""
    if getattr(update, "callback_query", None):
        asyncio.create_task(update.callback_query.answer())
        asyncio.create_task(update.callback_query.message.edit_text(text, reply_markup=reply_markup))
    else:
        asyncio.create_task(update.message.reply_text(text, reply_markup=reply_markup))


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update, context.bot_data.get("admin_id")):
        return
    text = await context.bot_data["controller"].start()
    await update.message.reply_text(text)


async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update, context.bot_data.get("admin_id")):
        return
    text = await context.bot_data["controller"].stop()
    await update.message.reply_text(text)


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update, context.bot_data.get("admin_id")):
        return
    text = await context.bot_data["controller"].status()
    await update.message.reply_text(text)


async def strategies_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update, context.bot_data.get("admin_id")):
        return
    text = await context.bot_data["controller"].strategies()
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Refresh", callback_data="strategies")]]
    )
    await _reply_or_edit(update, text, keyboard)


async def positions_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update, context.bot_data.get("admin_id")):
        return
    text = await context.bot_data["controller"].positions()
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Refresh", callback_data="positions")]]
    )
    await _reply_or_edit(update, text, keyboard)


async def logs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update, context.bot_data.get("admin_id")):
        return
    text = await context.bot_data["controller"].logs()
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Refresh", callback_data="logs")]]
    )
    await _reply_or_edit(update, text, keyboard)


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin(update, context.bot_data.get("admin_id")):
        return
    text = await context.bot_data["controller"].settings()
    await update.message.reply_text(text)

