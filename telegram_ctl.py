from __future__ import annotations

import json
import re
import time
from typing import Dict, Tuple

from crypto_bot.utils.logger import LOG_DIR


try:  # pragma: no cover - optional dependency
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import (
        ApplicationBuilder,
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
    )
except Exception:  # pragma: no cover - telegram not installed
    InlineKeyboardButton = InlineKeyboardMarkup = Update = object  # type: ignore
    ApplicationBuilder = CallbackQueryHandler = CommandHandler = ContextTypes = object  # type: ignore

LOG_FILE = LOG_DIR / "bot.log"
STRATEGY_FILE = LOG_DIR / "strategy_scores.json"
POSITIONS_FILE = LOG_DIR / "positions.log"

callback_timeout = 300
callback_state: Dict[str, Dict[str, Tuple[int, float]]] = {}
ITEMS_PER_PAGE = 5

POS_PATTERN = re.compile(
    r"Active (?P<symbol>\S+) (?P<side>\w+) (?P<amount>[0-9.]+) "
    r"entry (?P<entry>[0-9.]+) current (?P<current>[0-9.]+) "
    r"pnl \$(?P<pnl>[0-9.+-]+).*balance \$(?P<balance>[0-9.]+)"
)


def _cleanup(chat_id: str) -> None:
    now = time.time()
    store = callback_state.get(chat_id)
    if not store:
        return
    for key in list(store.keys()):
        _, ts = store[key]
        if now - ts > callback_timeout:
            del store[key]
    if not store:
        callback_state.pop(chat_id, None)


def set_page(chat_id: str | int, key: str, value: int) -> None:
    cid = str(chat_id)
    _cleanup(cid)
    store = callback_state.setdefault(cid, {})
    store[key] = (value, time.time())


def get_page(chat_id: str | int, key: str) -> int:
    cid = str(chat_id)
    _cleanup(cid)
    store = callback_state.get(cid)
    if not store:
        return 0
    val = store.get(key)
    if val is None:
        return 0
    page, ts = val
    if time.time() - ts > callback_timeout:
        del store[key]
        if not store:
            callback_state.pop(cid, None)
        return 0
    return page


def _paginate(lines: list[str], page: int) -> tuple[str, InlineKeyboardMarkup]:
    total_pages = max(1, (len(lines) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    page = max(0, min(page, total_pages - 1))
    start = page * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    page_lines = lines[start:end]
    keyboard = []
    if total_pages > 1:
        buttons = []
        if page > 0:
            buttons.append(InlineKeyboardButton("Prev", callback_data="prev"))
        if page < total_pages - 1:
            buttons.append(InlineKeyboardButton("Next", callback_data="next"))
        keyboard.append(buttons)
    markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    return "\n".join(page_lines) or "(empty)", markup


async def logs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    page = get_page(chat_id, "logs")
    if update.callback_query:
        if update.callback_query.data == "next":
            page += 1
        elif update.callback_query.data == "prev":
            page = max(0, page - 1)
        await update.callback_query.answer()
    else:
        page = 0
    set_page(chat_id, "logs", page)
    lines = []
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().splitlines()
        lines = lines[-100:]
    text, markup = _paginate(lines, page)
    if update.callback_query:
        await update.callback_query.message.edit_text(text, reply_markup=markup)
    else:
        await update.message.reply_text(text, reply_markup=markup)


async def strategies_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    page = get_page(chat_id, "strategies")
    if update.callback_query:
        if update.callback_query.data == "next":
            page += 1
        elif update.callback_query.data == "prev":
            page = max(0, page - 1)
        await update.callback_query.answer()
    else:
        page = 0
    set_page(chat_id, "strategies", page)
    lines = []
    if STRATEGY_FILE.exists():
        try:
            data = json.loads(STRATEGY_FILE.read_text())
            lines = [f"{k}: {v}" for k, v in data.items()]
        except Exception:
            lines = ["Invalid strategy file"]
    text, markup = _paginate(lines, page)
    if update.callback_query:
        await update.callback_query.message.edit_text(text, reply_markup=markup)
    else:
        await update.message.reply_text(text, reply_markup=markup)


async def positions_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    page = get_page(chat_id, "positions")
    if update.callback_query:
        if update.callback_query.data == "next":
            page += 1
        elif update.callback_query.data == "prev":
            page = max(0, page - 1)
        await update.callback_query.answer()
    else:
        page = 0
    set_page(chat_id, "positions", page)
    lines = []
    if POSITIONS_FILE.exists():
        for line in POSITIONS_FILE.read_text().splitlines():
            m = POS_PATTERN.search(line)
            if m:
                lines.append(
                    f"{m.group('symbol')} {m.group('side')} {m.group('amount')} pnl ${m.group('pnl')}"
                )
    text, markup = _paginate(lines, page)
    if update.callback_query:
        await update.callback_query.message.edit_text(text, reply_markup=markup)
    else:
        await update.message.reply_text(text, reply_markup=markup)


def main(token: str) -> None:
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("logs", logs_cmd))
    app.add_handler(CommandHandler("strategies", strategies_cmd))
    app.add_handler(CommandHandler("positions", positions_cmd))
    app.add_handler(CallbackQueryHandler(logs_cmd, pattern="^(next|prev)$"))
    app.add_handler(CallbackQueryHandler(strategies_cmd, pattern="^(next|prev)$"))
    app.add_handler(CallbackQueryHandler(positions_cmd, pattern="^(next|prev)$"))
    app.run_polling()


if __name__ == "__main__":
    import os

    token = os.environ.get("TELEGRAM_TOKEN", "")
    if not token:
        raise SystemExit("Set TELEGRAM_TOKEN")
    main(token)
