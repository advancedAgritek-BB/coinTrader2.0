from __future__ import annotations

import asyncio
import threading
import time
import json
import yaml
import contextlib
from pathlib import Path
from typing import Dict


import schedule

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest, NetworkError
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)
from crypto_bot.telegram_ctl import _paginate, get_page, set_page

from crypto_bot.portfolio_rotator import PortfolioRotator
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.telegram import TelegramNotifier, is_admin

from crypto_bot import log_reader, console_monitor
from .telegram_ctl import BotController

START = "START"
STOP = "STOP"
STATUS = "STATUS"
LOG = "LOG"
ROTATE = "ROTATE"
TOGGLE = "TOGGLE"
MENU = "MENU"
RELOAD = "RELOAD"
SIGNALS = "SIGNALS"
BALANCE = "BALANCE"
TRADES = "TRADES"
TRADE_HISTORY = "TRADE_HISTORY"
PANIC_SELL = "PANIC_SELL"
CONFIG = "CONFIG"
EDIT_TRADE_SIZE = "EDIT_TRADE_SIZE"
EDIT_MAX_TRADES = "EDIT_MAX_TRADES"
EDIT_VALUE = 0
PNL_STATS = "PNL_STATS"

ASSET_SCORES_FILE = LOG_DIR / "asset_scores.json"
SIGNALS_FILE = LOG_DIR / "asset_scores.json"
TRADES_FILE = LOG_DIR / "trades.csv"
CONFIG_FILE = Path("crypto_bot/config.yaml")

# Text sent via ``TelegramNotifier`` when the bot starts.
MENU_TEXT = "Select a command:"


def _back_to_menu_markup() -> InlineKeyboardMarkup:
    """Return a markup with a single 'Back to Menu' button."""
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Back to Menu", callback_data=MENU)]]
    )


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
        command_cooldown: float = 5,
    ) -> None:
        self.notifier = notifier
        self.token = notifier.token
        self.chat_id = notifier.chat_id
        self.state = state
        self.log_file = Path(log_file)
        self.rotator = rotator
        self.controller = BotController(state, exchange, log_file=self.log_file, trades_file=TRADES_FILE)
        self.exchange = exchange
        self.wallet = wallet
        self.command_cooldown = command_cooldown
        self._last_exec: Dict[tuple[str, str], float] = {}
        self.logger = setup_logger(__name__, LOG_DIR / "telegram_ui.log")

        self.app = ApplicationBuilder().token(self.token).build()
        if hasattr(self.app, "bot_data"):
            self.app.bot_data["controller"] = self.controller
            self.app.bot_data["admin_id"] = self.chat_id
        self.app.add_handler(CommandHandler("start", self.start_cmd))
        self.app.add_handler(CommandHandler("stop", self.stop_cmd))
        self.app.add_handler(CommandHandler("status", self.status_cmd))
        self.app.add_handler(CommandHandler("log", self.log_cmd))
        self.app.add_handler(CommandHandler("rotate_now", self.rotate_now_cmd))
        self.app.add_handler(CommandHandler("toggle_mode", self.toggle_mode_cmd))
        self.app.add_handler(CommandHandler("reload", self.reload_cmd))
        self.app.add_handler(CommandHandler("menu", self.menu_cmd))
        self.app.add_handler(CommandHandler("signals", self.show_signals))
        self.app.add_handler(CommandHandler("balance", self.show_balance))
        self.app.add_handler(CommandHandler("trades", self.show_trades))
        self.app.add_handler(CommandHandler("trade_history", self.show_trade_history))
        self.app.add_handler(CommandHandler("config", self.show_config))
        self.app.add_handler(CommandHandler("pnl_stats", self.show_pnl_stats))
        self.app.add_handler(CommandHandler("panic_sell", self.panic_sell_cmd))
        self.app.add_handler(CallbackQueryHandler(self.start_cmd, pattern=f"^{START}$"))
        self.app.add_handler(CallbackQueryHandler(self.stop_cmd, pattern=f"^{STOP}$"))
        self.app.add_handler(CallbackQueryHandler(self.status_cmd, pattern=f"^{STATUS}$"))
        self.app.add_handler(CallbackQueryHandler(self.log_cmd, pattern=f"^{LOG}$"))
        self.app.add_handler(CallbackQueryHandler(self.rotate_now_cmd, pattern=f"^{ROTATE}$"))
        self.app.add_handler(CallbackQueryHandler(self.toggle_mode_cmd, pattern=f"^{TOGGLE}$"))
        self.app.add_handler(CallbackQueryHandler(self.reload_cmd, pattern=f"^{RELOAD}$"))
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
        self.app.add_handler(
            CallbackQueryHandler(self.show_trade_history, pattern=f"^{TRADE_HISTORY}$")
        )
        self.app.add_handler(
            CallbackQueryHandler(self.show_trade_history, pattern="^(next|prev)$")
        )
        self.app.add_handler(
            CallbackQueryHandler(self.panic_sell_cmd, pattern=f"^{PANIC_SELL}$")
        )
        self.app.add_handler(
            CallbackQueryHandler(self.show_config, pattern=f"^{CONFIG}$")
        )

        conv = ConversationHandler(
            entry_points=[
                CallbackQueryHandler(self.edit_trade_size, pattern=f"^{EDIT_TRADE_SIZE}$"),
                CallbackQueryHandler(self.edit_max_trades, pattern=f"^{EDIT_MAX_TRADES}$"),
            ],
            states={
                EDIT_VALUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.set_config_value)]
            },
            fallbacks=[],
            per_message=False,
        )
        self.app.add_handler(conv)
        self.app.add_handler(
            CallbackQueryHandler(self.show_pnl_stats, pattern=f"^{PNL_STATS}$")
        )

        self.scheduler_thread: threading.Thread | None = None

        schedule.every().day.at("00:00").do(self.send_daily_summary)
        self.scheduler_thread = threading.Thread(
            target=self._run_scheduler, daemon=True
        )
        self.scheduler_thread.start()

        self.task: asyncio.Task | None = None

    def run_async(self) -> asyncio.Task:
        """Start polling within the current event loop with retry/backoff."""

        async def run() -> None:
            await self.app.initialize()
            self.notifier.notify(MENU_TEXT)
            backoff = 1
            last_log = 0
            try:
                while True:
                    try:
                        await self.app.start()
                        await self.app.updater.start_polling()
                        await self.app.updater.wait_closed()
                        backoff = 1
                        last_log = 0
                    except (NetworkError, OSError) as exc:
                        if backoff != last_log:
                            self.logger.error(
                                "Telegram polling error: %s; retrying in %ss",
                                exc,
                                backoff,
                            )
                            last_log = backoff
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                    except Exception as exc:  # pragma: no cover - unexpected
                        if backoff != last_log:
                            self.logger.error(
                                "Telegram poller failed: %s; retrying in %ss",
                                exc,
                                backoff,
                            )
                            last_log = backoff
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                    finally:
                        with contextlib.suppress(Exception):
                            await self.app.updater.stop()
                        with contextlib.suppress(Exception):
                            await self.app.stop()
            except asyncio.CancelledError:
                pass
            finally:
                with contextlib.suppress(Exception):
                    await self.app.shutdown()

        self.task = asyncio.create_task(run())
        return self.task

    def _get_chat_id(self, update: Update) -> str:
        if getattr(update, "effective_chat", None):
            return str(update.effective_chat.id)
        if getattr(update, "message", None) and getattr(update.message, "chat_id", None):
            return str(update.message.chat_id)
        if getattr(update, "callback_query", None):
            msg = update.callback_query.message
            if getattr(msg, "chat_id", None):
                return str(msg.chat_id)
        return str(self.chat_id)

    async def _reply(
        self,
        update: Update,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> None:
        if getattr(update, "callback_query", None):
            try:
                await update.callback_query.message.edit_text(text, reply_markup=reply_markup)
            except BadRequest as exc:
                if "Message is not modified" not in str(exc):
                    raise
        else:
            await update.message.reply_text(text, reply_markup=reply_markup)

    async def _check_cooldown(self, update: Update, command: str) -> bool:
        chat = self._get_chat_id(update)
        now = time.time()
        key = (chat, command)
        last = self._last_exec.get(key)
        if last is not None and now - last < self.command_cooldown:
            await self._reply(update, "Please wait")
            return False
        self._last_exec[key] = now
        return True

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

    async def _check_admin(self, update: Update) -> bool:
        """Verify the update came from an authorized chat."""
        chat_id = str(getattr(getattr(update, "effective_chat", None), "id", ""))
        if not is_admin(chat_id):
            if getattr(update, "message", None):
                await update.message.reply_text("Unauthorized")
            elif getattr(update, "callback_query", None):
                await update.callback_query.answer("Unauthorized", show_alert=True)
            self.logger.warning("Ignoring unauthorized command from %s", chat_id)
            return False
        return True

    # Command handlers -------------------------------------------------
    async def start_cmd(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_cooldown(update, "start"):
            return
        if not await self._check_admin(update):
            return
        text = await self.controller.start()
        await update.message.reply_text(text)
        self.state["running"] = True
        await self._reply(update, "Trading started", reply_markup=_back_to_menu_markup())
        await self.menu_cmd(update, context)

    async def stop_cmd(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_cooldown(update, "stop"):
            return
        if not await self._check_admin(update):
            return
        text = await self.controller.stop()
        await update.message.reply_text(text)
        self.state["running"] = False
        await self._reply(update, "Trading stopped", reply_markup=_back_to_menu_markup())

    async def status_cmd(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_cooldown(update, "status"):
            return
        text = await self.controller.status()
        await self._reply(update, text, reply_markup=_back_to_menu_markup())


    async def log_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_cooldown(update, "log"):
            return
        if not await self._check_admin(update):
            return
        if self.log_file.exists():
            lines = self.log_file.read_text().splitlines()[-20:]
            text = "\n".join(lines) if lines else "(no logs)"
        else:
            text = "Log file not found"
        await self._reply(update, text, reply_markup=_back_to_menu_markup())

    async def rotate_now_cmd(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_cooldown(update, "rotate_now"):
            return
        if not await self._check_admin(update):
            return
        if not (self.rotator and self.exchange and self.wallet):
            await self._reply(update, "Rotation not configured", reply_markup=_back_to_menu_markup())
            return
        try:
            if asyncio.iscoroutinefunction(getattr(self.exchange, "fetch_balance", None)):
                bal = await self.exchange.fetch_balance()
            else:
                bal = await asyncio.to_thread(self.exchange.fetch_balance)
            holdings = {
                k: (v.get("total") if isinstance(v, dict) else v)
                for k, v in bal.items()
            }
            await self.rotator.rotate(
                self.exchange,
                self.wallet,
                holdings,
            )
            await self._reply(update, "Portfolio rotated", reply_markup=_back_to_menu_markup())
        except Exception as exc:  # pragma: no cover - network
            self.logger.error("Rotation failed: %s", exc)
            await self._reply(update, "Rotation failed", reply_markup=_back_to_menu_markup())

    def send_daily_summary(self) -> None:
        stats = log_reader.trade_summary(str(LOG_DIR / "trades.csv"))
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
        if not await self._check_cooldown(update, "toggle_mode"):
            return
        if not await self._check_admin(update):
            return
        mode = self.state.get("mode")
        mode = "onchain" if mode == "cex" else "cex"
        self.state["mode"] = mode
        await self._reply(update, f"Mode set to {mode}", reply_markup=_back_to_menu_markup())

    async def reload_cmd(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_cooldown(update, "reload"):
            return
        if not await self._check_admin(update):
            return
        await self.controller.reload_config()
        await self._reply(update, "Config reload scheduled", reply_markup=_back_to_menu_markup())

    async def panic_sell_cmd(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_cooldown(update, "panic_sell"):
            return
        if not await self._check_admin(update):
            return
        text = await self.controller.close_all_positions()
        await self._reply(update, text, reply_markup=_back_to_menu_markup())

    async def menu_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_cooldown(update, "menu"):
            return
        if not await self._check_admin(update):
            return
        keyboard = [
            [
                InlineKeyboardButton("Start", callback_data=START),
                InlineKeyboardButton("Stop", callback_data=STOP),
                InlineKeyboardButton("Status", callback_data=STATUS),
            ],
            [
                InlineKeyboardButton("Log", callback_data=LOG),
                InlineKeyboardButton("Rotate Now", callback_data=ROTATE),
                InlineKeyboardButton("Toggle Mode", callback_data=TOGGLE),
                InlineKeyboardButton("Reload", callback_data=RELOAD),
                InlineKeyboardButton("Panic Sell", callback_data=PANIC_SELL),
            ],
            [
                InlineKeyboardButton("Signals", callback_data=SIGNALS),
                InlineKeyboardButton("Balance", callback_data=BALANCE),
                InlineKeyboardButton("Trades", callback_data=TRADES),
                InlineKeyboardButton("Trade History", callback_data=TRADE_HISTORY),
                InlineKeyboardButton("PnL Stats", callback_data=PNL_STATS),
            ],
            [
                InlineKeyboardButton("Config Settings", callback_data=CONFIG),
            ],
        ]
        markup = InlineKeyboardMarkup(keyboard)
        await self._reply(update, "Select a command:", reply_markup=markup)

    async def show_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_cooldown(update, "signals"):
            return
        if not await self._check_admin(update):
            return
        # Use ``ASSET_SCORES_FILE`` so tests can patch the path easily.
        if ASSET_SCORES_FILE.exists():
            try:
                data = json.loads(ASSET_SCORES_FILE.read_text())
                lines = [f"{k}: {v:.2f}" for k, v in data.items()]
                text = "\n".join(lines) if lines else "(no signals)"
            except Exception:
                text = "Invalid signals file"
        else:
            text = "No signals found"
        await self._reply(update, text, reply_markup=_back_to_menu_markup())

    async def show_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_cooldown(update, "balance"):
            return
        if not await self._check_admin(update):
            return
        if not self.exchange:
            await self._reply(update, "Exchange not configured", reply_markup=_back_to_menu_markup())
            return
        try:
            if asyncio.iscoroutinefunction(getattr(self.exchange, "fetch_balance", None)):
                bal = await self.exchange.fetch_balance()
            else:
                bal = await asyncio.to_thread(self.exchange.fetch_balance)
            free_usdt = (
                bal.get("USDT", {}).get("free")
                if isinstance(bal.get("USDT"), dict)
                else None
            )
            lines = [f"Free USDT: {free_usdt or 0}"]
            lines += [
                f"{k}: {v.get('total') if isinstance(v, dict) else v}"
                for k, v in bal.items()
            ]
            text = "\n".join(lines) if lines else "(no balance)"
        except Exception as exc:  # pragma: no cover - network
            self.logger.error("Balance fetch failed: %s", exc)
            text = "Balance fetch failed"
        await self._reply(update, text, reply_markup=_back_to_menu_markup())

    async def show_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_cooldown(update, "trades"):
            return
        if not await self._check_admin(update):
            return
        if TRADES_FILE.exists():
            lines = await console_monitor.trade_stats_lines(self.exchange, TRADES_FILE)
            text = "\n".join(lines) if lines else "(no trades)"
        else:
            text = "No trades found"
        await self._reply(update, text, reply_markup=_back_to_menu_markup())

    async def show_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_cooldown(update, "config"):
            return
        if not await self._check_admin(update):
            return
        cfg = {}
        if CONFIG_FILE.exists():
            try:
                cfg = yaml.safe_load(CONFIG_FILE.read_text()) or {}
            except Exception:
                cfg = {}
        text = (
            f"trade_size_pct: {cfg.get('trade_size_pct')}\n"
            f"max_open_trades: {cfg.get('max_open_trades')}"
        )
        keyboard = [
            [InlineKeyboardButton("Edit Trade Size %", callback_data=EDIT_TRADE_SIZE)],
            [InlineKeyboardButton("Edit Max Open Trades", callback_data=EDIT_MAX_TRADES)],
        ]
        markup = InlineKeyboardMarkup(keyboard)
        await self._reply(update, text, reply_markup=markup)

    async def edit_trade_size(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if not await self._check_admin(update):
            return ConversationHandler.END
        context.user_data["config_key"] = "trade_size_pct"
        await self._reply(update, "Enter trade size percentage (0-1):")
        return EDIT_VALUE

    async def edit_max_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if not await self._check_admin(update):
            return ConversationHandler.END
        context.user_data["config_key"] = "max_open_trades"
        await self._reply(update, "Enter max open trades (integer):")
        return EDIT_VALUE

    async def set_config_value(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        if not await self._check_admin(update):
            return ConversationHandler.END
        key = context.user_data.get("config_key")
        if not key:
            return ConversationHandler.END
        value_text = update.message.text if update.message else ""
        try:
            if key == "trade_size_pct":
                val = float(value_text)
                if not 0 < val <= 1:
                    raise ValueError
            else:
                val = int(value_text)
                if val <= 0:
                    raise ValueError
        except ValueError:
            await self._reply(update, "Invalid value, try again:")
            return EDIT_VALUE
        cfg = {}
        if CONFIG_FILE.exists():
            try:
                cfg = yaml.safe_load(CONFIG_FILE.read_text()) or {}
            except Exception:
                cfg = {}
        cfg[key] = val
        CONFIG_FILE.write_text(json.dumps(cfg))
        await self.controller.reload_config()
        await self._reply(update, f"{key} updated to {val}")
        return ConversationHandler.END
    async def show_pnl_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._check_cooldown(update, "pnl_stats"):
            return
        if not await self._check_admin(update):
            return
        if TRADES_FILE.exists():
            stats = log_reader.trade_summary(TRADES_FILE)
            text = (
                f"Total PnL: {stats['total_pnl']:.2f}\n"
                f"Win rate: {stats['win_rate']*100:.1f}%\n"
                f"Active positions: {stats['active_positions']}"
            )
        else:
            text = "No trades found"
        await self._reply(update, text)

    async def show_trade_history(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not await self._check_cooldown(update, "trade_history"):
            return
        if not await self._check_admin(update):
            return
        chat_id = self._get_chat_id(update)
        page = get_page(chat_id, "trade_history")
        if getattr(update, "callback_query", None):
            if update.callback_query.data == "next":
                page += 1
            elif update.callback_query.data == "prev":
                page = max(0, page - 1)
            else:
                page = 0
            await update.callback_query.answer()
        else:
            page = 0
        set_page(chat_id, "trade_history", page)
        lines: list[str] = []
        if TRADES_FILE.exists():
            lines = TRADES_FILE.read_text().splitlines()[-100:]
        text, markup = _paginate(lines, page)
        await self._reply(update, text, reply_markup=markup)
