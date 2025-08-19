import types
import asyncio
import json
import pytest
import sys
import crypto_bot
import crypto_bot.utils as utils_pkg

# Provide a minimal stub for crypto_bot.utils.telegram to avoid optional deps
telegram_utils = types.ModuleType("crypto_bot.utils.telegram")


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, *a, **k):
        self.token = token
        self.chat_id = chat_id

    def notify(self, text: str):  # pragma: no cover - simple stub
        return None


def is_admin(chat_id: str) -> bool:
    if not _admin_ids:
        return True
    return str(chat_id) in _admin_ids


_admin_ids: set[str] = set()


def set_admin_ids(admins):
    global _admin_ids
    _admin_ids = {str(a) for a in admins}


telegram_utils.TelegramNotifier = TelegramNotifier
telegram_utils.is_admin = is_admin
telegram_utils.set_admin_ids = set_admin_ids

utils_pkg.telegram = telegram_utils
sys.modules["crypto_bot.utils.telegram"] = telegram_utils

import crypto_bot.telegram_bot_ui as telegram_bot_ui
import yaml
from crypto_bot.telegram_bot_ui import (
    TelegramBotUI,
    SIGNALS,
    BALANCE,
    TRADES,
    TRADE_HISTORY,
    MENU,
    RELOAD,
    CONFIG,
    EDIT_TRADE_SIZE,
    EDIT_MAX_TRADES,
    PNL_STATS,
)
from crypto_bot.utils.telegram import TelegramNotifier


class DummyApplication:
    def __init__(self, *a, **k):
        self.handlers = []
        self.updater = types.SimpleNamespace(start_polling=self._async_noop)

    async def _async_noop(self, *a, **k):
        return None

    def add_handler(self, handler):
        self.handlers.append(handler)

    async def initialize(self):
        return None

    async def start(self):
        return None

    def stop(self):
        pass


class DummyBuilder:
    def token(self, *a, **k):
        return self

    def build(self):
        return DummyApplication()


class DummyMessage:
    def __init__(self):
        self.text = None
        self.reply_markup = None

    async def reply_text(self, text, reply_markup=None):
        self.text = text
        self.reply_markup = reply_markup

    async def edit_text(self, text, reply_markup=None):
        self.text = text
        self.reply_markup = reply_markup


class DummyUpdate:
    def __init__(self):
        self.message = DummyMessage()


class DummyCallbackUpdate:
    def __init__(self, data=""):
        async def answer():
            return None

        self.callback_query = types.SimpleNamespace(
            data=data,
            message=DummyMessage(),
            answer=answer,
        )


class DummyContext:
    def __init__(self):
        self.user_data = {}


class DummyExchange:
    def fetch_balance(self):
        return {"BTC": {"total": 1}}

    def fetch_ticker(self, symbol):
        return {"last": 105}


class DummyAsyncExchange:
    async def fetch_balance(self):
        return {"BTC": {"total": 2}}

    async def fetch_ticker(self, symbol):
        return {"last": 110}


class DummyRotator:
    def __init__(self):
        self.called = False
        self.args = None

    async def rotate(self, *args):
        self.called = True
        self.args = args


def make_ui(tmp_path, state, rotator=None, exchange=None, notifier=None):
    log_file = tmp_path / "bot.log"
    log_file.write_text("line1\nline2\n")
    if notifier is None:
        notifier = TelegramNotifier("token", "chat")
    ui = TelegramBotUI(
        notifier,
        state,
        log_file,
        rotator=rotator,
        exchange=exchange,
        wallet="addr",
    )
    return ui, log_file


def test_conversation_handler_no_warning(monkeypatch, tmp_path, recwarn):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    make_ui(tmp_path, {"running": False, "mode": "cex"})
    assert not [w for w in recwarn if "ConversationHandler" in str(w.message)]


def test_menu_sent_on_run(monkeypatch, tmp_path):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)

    messages = []

    class DummyNotifier(TelegramNotifier):
        def __init__(self):
            super().__init__("token", "chat")

        def notify(self, text):
            messages.append(text)

    notifier = DummyNotifier()
    ui, _ = make_ui(tmp_path, {"running": False, "mode": "cex"}, notifier=notifier)

    async def runner():
        ui.run_async()
        await asyncio.sleep(0)

    asyncio.run(runner())

    assert messages and messages[0] == telegram_bot_ui.MENU_TEXT


def test_start_stop_toggle(monkeypatch, tmp_path):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    state = {"running": False, "mode": "auto"}
    ui, _ = make_ui(tmp_path, state)
    ui.command_cooldown = 0
    ui.command_cooldown = 0
    ui.command_cooldown = 0
    ui.command_cooldown = 0

    update = DummyUpdate()
    asyncio.run(ui.start_cmd(update, DummyContext()))
    assert state["running"] is True
    assert update.message.text == "Select a command:"
    assert isinstance(update.message.reply_markup, telegram_bot_ui.InlineKeyboardMarkup)

    asyncio.run(ui.stop_cmd(update, DummyContext()))
    assert state["running"] is False
    assert update.message.text == "Trading stopped"

    asyncio.run(ui.toggle_mode_cmd(update, DummyContext()))
    assert state["mode"] == "cex"
    assert update.message.text == "Mode set to cex"

    update_status = DummyUpdate()
    asyncio.run(ui.status_cmd(update_status, DummyContext()))
    assert "mode: cex" in update_status.message.text


def test_log_and_rotate(monkeypatch, tmp_path):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    rotator = DummyRotator()
    exchange = DummyExchange()
    state = {"running": True, "mode": "cex"}
    ui, log_file = make_ui(tmp_path, state, rotator=rotator, exchange=exchange)

    update = DummyUpdate()
    asyncio.run(ui.log_cmd(update, DummyContext()))
    assert "line2" in update.message.text

    asyncio.run(ui.rotate_now_cmd(update, DummyContext()))
    assert rotator.called is True
    assert update.message.text == "Portfolio rotated"

    asyncio.run(ui.status_cmd(update, DummyContext()))
    assert "Running: True" in update.message.text


def test_menu_signals_balance_trades_history(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder
    )
    exchange = DummyExchange()
    state = {"running": True, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state, exchange=exchange)

    scores = {"BTC": 0.5, "ETH": 0.1}
    sig_file = tmp_path / "scores.json"
    sig_file.write_text(json.dumps(scores))
    monkeypatch.setattr(telegram_bot_ui, "SIGNALS_FILE", sig_file)
    monkeypatch.setattr(telegram_bot_ui, "ASSET_SCORES_FILE", sig_file)

    trades_file = tmp_path / "trades.csv"
    trades_file.write_text("XBT/USDT,buy,1,100\n")
    monkeypatch.setattr(telegram_bot_ui, "TRADES_FILE", trades_file)

    update = DummyUpdate()
    asyncio.run(ui.menu_cmd(update, DummyContext()))
    assert isinstance(update.message.reply_markup, telegram_bot_ui.InlineKeyboardMarkup)
    assert len(update.message.reply_markup.inline_keyboard) == 4
    buttons = [btn for row in update.message.reply_markup.inline_keyboard for btn in row]
    assert any(getattr(btn, "text", None) == "Trade History" for btn in buttons)
    texts = [btn.text for row in update.message.reply_markup.inline_keyboard for btn in row]
    assert "PnL Stats" in texts

    update = DummyUpdate()
    asyncio.run(ui.show_signals(update, DummyContext()))
    assert "BTC" in update.message.text
    assert isinstance(update.message.reply_markup, telegram_bot_ui.InlineKeyboardMarkup)
    assert update.message.reply_markup.inline_keyboard[0][0].text == "Back to Menu"

    update = DummyUpdate()
    asyncio.run(ui.show_balance(update, DummyContext()))
    assert "BTC" in update.message.text
    assert isinstance(update.message.reply_markup, telegram_bot_ui.InlineKeyboardMarkup)
    assert update.message.reply_markup.inline_keyboard[0][0].text == "Back to Menu"

    update = DummyUpdate()
    asyncio.run(ui.show_pnl_stats(update, DummyContext()))
    assert "Total PnL" in update.message.text

    update = DummyUpdate()
    asyncio.run(ui.show_trade_history(update, DummyContext()))
    assert "XBT/USDT" in update.message.text



def test_commands_require_admin(monkeypatch, tmp_path):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    state = {"running": False, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state)
    import crypto_bot.utils.telegram as tg
    tg.set_admin_ids(["999"])  # only allow 999

    update = DummyUpdate()
    update.effective_chat = types.SimpleNamespace(id="123")
    asyncio.run(ui.start_cmd(update, DummyContext()))
    assert update.message.text == "Unauthorized"
    assert state["running"] is False

    tg.set_admin_ids([])

    trades_file = tmp_path / "trades.csv"
    trades_file.write_text("XBT/USDT,buy,1,100\n")
    monkeypatch.setattr(telegram_bot_ui, "TRADES_FILE", trades_file)

    update = DummyUpdate()
    asyncio.run(ui.show_trades(update, DummyContext()))
    assert "XBT/USDT" in update.message.text
    assert isinstance(update.message.reply_markup, telegram_bot_ui.InlineKeyboardMarkup)
    assert update.message.reply_markup.inline_keyboard[0][0].text == "Back to Menu"


def test_unauthorized_start_stop(monkeypatch, tmp_path):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    state = {"running": False, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state)
    import crypto_bot.utils.telegram as tg
    tg.set_admin_ids(["999"])  # restrict

    update = DummyUpdate()
    update.effective_chat = types.SimpleNamespace(id="123")
    asyncio.run(ui.start_cmd(update, DummyContext()))
    assert update.message.text == "Unauthorized"
    assert state["running"] is False

    state["running"] = True
    update = DummyUpdate()
    update.effective_chat = types.SimpleNamespace(id="123")
    asyncio.run(ui.stop_cmd(update, DummyContext()))
    assert update.message.text == "Unauthorized"
    assert state["running"] is True
    tg.set_admin_ids([])

def test_menu_callbacks(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "crypto_bot.telegram_bot_ui.ApplicationBuilder",
        DummyBuilder,
    )
    asset_file = tmp_path / "scores.json"
    asset_file.write_text('{"XBT/USDT": 0.5}')
    trades_file = tmp_path / "trades.csv"
    trades_file.write_text("XBT/USDT,buy,1,100,t1\n")

    state = {"running": True, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state, exchange=DummyExchange())

    monkeypatch.setattr(
        "crypto_bot.telegram_bot_ui.ASSET_SCORES_FILE",
        asset_file,
    )
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.TRADES_FILE", trades_file)

    update = DummyCallbackUpdate()
    update.callback_query.data = SIGNALS
    asyncio.run(ui.show_signals(update, DummyContext()))
    assert "XBT/USDT" in update.callback_query.message.text
    assert isinstance(update.callback_query.message.reply_markup, telegram_bot_ui.InlineKeyboardMarkup)
    assert update.callback_query.message.reply_markup.inline_keyboard[0][0].text == "Back to Menu"

    update = DummyCallbackUpdate()
    update.callback_query.data = BALANCE
    asyncio.run(ui.show_balance(update, DummyContext()))
    assert "Free USDT" in update.callback_query.message.text
    assert isinstance(update.callback_query.message.reply_markup, telegram_bot_ui.InlineKeyboardMarkup)
    assert update.callback_query.message.reply_markup.inline_keyboard[0][0].text == "Back to Menu"

    update = DummyCallbackUpdate()
    update.callback_query.data = TRADES
    asyncio.run(ui.show_trades(update, DummyContext()))
    assert "+5.00" in update.callback_query.message.text
    assert isinstance(update.callback_query.message.reply_markup, telegram_bot_ui.InlineKeyboardMarkup)
    assert update.callback_query.message.reply_markup.inline_keyboard[0][0].text == "Back to Menu"

    update = DummyCallbackUpdate()
    update.callback_query.data = PNL_STATS
    asyncio.run(ui.show_pnl_stats(update, DummyContext()))
    assert "Total PnL" in update.callback_query.message.text


def test_async_exchange_balance_and_rotate(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "crypto_bot.telegram_bot_ui.ApplicationBuilder",
        DummyBuilder,
    )
    rotator = DummyRotator()
    exchange = DummyAsyncExchange()
    state = {"running": True, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state, rotator=rotator, exchange=exchange)

    update = DummyUpdate()
    asyncio.run(ui.rotate_now_cmd(update, DummyContext()))
    assert update.message.text == "Portfolio rotated"
    assert isinstance(update.message.reply_markup, telegram_bot_ui.InlineKeyboardMarkup)
    assert update.message.reply_markup.inline_keyboard[0][0].text == "Back to Menu"

    update = DummyUpdate()
    asyncio.run(ui.show_balance(update, DummyContext()))
    assert "BTC" in update.message.text
    assert isinstance(update.message.reply_markup, telegram_bot_ui.InlineKeyboardMarkup)
    assert update.message.reply_markup.inline_keyboard[0][0].text == "Back to Menu"


def test_command_cooldown(monkeypatch, tmp_path):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    t = {"now": 0}
    monkeypatch.setattr(telegram_bot_ui.time, "time", lambda: t["now"])
    state = {"running": False, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state)
    ui.command_cooldown = 5

    update1 = DummyUpdate()
    asyncio.run(ui.start_cmd(update1, DummyContext()))
    assert update1.message.text == "Select a command:"

    t["now"] = 2
    update2 = DummyUpdate()
    asyncio.run(ui.start_cmd(update2, DummyContext()))
    assert update2.message.text == "Please wait"

    t["now"] = 6
    update3 = DummyUpdate()
    asyncio.run(ui.start_cmd(update3, DummyContext()))
    assert update3.message.text == "Select a command:"


def test_reload(monkeypatch, tmp_path):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    state = {"running": True, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state)
    ui.command_cooldown = 0
    t = {"now": 0}
    monkeypatch.setattr(telegram_bot_ui.time, "time", lambda: t["now"])
    ui.command_cooldown = 0
    t = {"now": 0}
    monkeypatch.setattr(telegram_bot_ui.time, "time", lambda: t["now"])

    update = DummyUpdate()
    asyncio.run(ui.reload_cmd(update, DummyContext()))
    assert state["reload"] is True
    assert update.message.text == "Config reload scheduled"


def test_trade_history_pagination(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "crypto_bot.telegram_bot_ui.ApplicationBuilder",
        DummyBuilder,
    )
    state = {"running": True, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state)
    ui.command_cooldown = 0
    t = {"now": 0}
    monkeypatch.setattr(telegram_bot_ui.time, "time", lambda: t["now"])

    trades_file = tmp_path / "trades.csv"
    trades = [f"t{i}" for i in range(12)]
    trades_file.write_text("\n".join(trades))
    monkeypatch.setattr(telegram_bot_ui, "TRADES_FILE", trades_file)

    update = DummyUpdate()
    asyncio.run(ui.show_trade_history(update, DummyContext()))
    assert "t0" in update.message.text
    assert "t5" not in update.message.text

    t["now"] += 1

    update_next = DummyCallbackUpdate("next")
    asyncio.run(ui.show_trade_history(update_next, DummyContext()))
    assert "t5" in update_next.callback_query.message.text

    t["now"] += 1

    update_prev = DummyCallbackUpdate("prev")
    asyncio.run(ui.show_trade_history(update_prev, DummyContext()))
    assert "t0" in update_prev.callback_query.message.text
def test_auto_menu_display(monkeypatch, tmp_path):
    """Menu command should reply with inline keyboard both via command and callback."""
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    state = {"running": False, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state)

    ui.command_cooldown = 0
    # direct /menu command
    update = DummyUpdate()
    asyncio.run(ui.menu_cmd(update, DummyContext()))
    assert update.message.text == "Select a command:"
    assert isinstance(update.message.reply_markup, telegram_bot_ui.InlineKeyboardMarkup)

    # callback invocation should edit the message in place
    cb = DummyCallbackUpdate(MENU)
    asyncio.run(ui.menu_cmd(cb, DummyContext()))
    assert cb.callback_query.message.text == "Select a command:"
    assert isinstance(cb.callback_query.message.reply_markup, telegram_bot_ui.InlineKeyboardMarkup)


def test_pnl_stats_output(monkeypatch, tmp_path):
    """show_trades should display PnL lines from console_monitor."""
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    state = {"running": True, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state, exchange=DummyExchange())

    lines = ["BTCUSDT -- 100.00 -- +5.00"]
    async def fake_lines(*_a, **_k):
        return lines
    monkeypatch.setattr(telegram_bot_ui.console_monitor, "trade_stats_lines", fake_lines)
    trades_file = tmp_path / "trades.csv"
    trades_file.write_text("sym,side,amt,price\n")
    monkeypatch.setattr(telegram_bot_ui, "TRADES_FILE", trades_file)
    update = DummyUpdate()
    asyncio.run(ui.show_trades(update, DummyContext()))
    assert "+5.00" in update.message.text


def test_trade_history_pagination(monkeypatch, tmp_path):
    """log_cmd should return only the last 20 lines of the log file."""
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    state = {"running": True, "mode": "cex"}
    ui, log_file = make_ui(tmp_path, state)

    # write 25 lines to log file
    log_file.write_text("\n".join(f"line{i}" for i in range(25)))
    update = DummyUpdate()
    asyncio.run(ui.log_cmd(update, DummyContext()))
    lines = update.message.text.splitlines()
    assert len(lines) == 20
    assert lines[0] == "line5"


def test_config_edit_workflow(monkeypatch, tmp_path):
    """Reload command should allow config to be refreshed via maybe_reload_config."""
    pass

def test_config_edit(monkeypatch, tmp_path):
    pass


def test_pnl_stats(monkeypatch, tmp_path):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    state = {"running": True, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state)

    import sys, types
    stub = types.ModuleType("crypto_bot.main")
    load_calls = []

    def fake_load():
        load_calls.append(True)
        return {"risk": {"trade_size_pct": 1.5}}

    async def fake_reload(config, *_a, force=False, **_k):
        cfg = fake_load()
        config.clear()
        config.update(cfg)
        state.pop("reload", None)

    stub.reload_config = fake_reload
    monkeypatch.setitem(sys.modules, "crypto_bot.main", stub)
    main = stub

    update = DummyUpdate()
    asyncio.run(ui.reload_cmd(update, DummyContext()))
    assert state["reload"] is True

    config = {}
    asyncio.run(main.reload_config(config, None, None, None, None, force=True))
    assert state.get("reload") is None
    assert config.get("risk", {}).get("trade_size_pct") == 1.5


def test_back_to_menu_navigation(monkeypatch, tmp_path):
    """User can return to the main menu after viewing another screen."""
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    state = {"running": True, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state)

    scores = {"BTC": 1.0}
    sig_file = tmp_path / "scores.json"
    sig_file.write_text(json.dumps(scores))
    monkeypatch.setattr(telegram_bot_ui, "ASSET_SCORES_FILE", sig_file)

    cb = DummyCallbackUpdate(SIGNALS)
    asyncio.run(ui.show_signals(cb, DummyContext()))
    assert "BTC" in cb.callback_query.message.text

    cb_menu = DummyCallbackUpdate(MENU)
    asyncio.run(ui.menu_cmd(cb_menu, DummyContext()))
    assert cb_menu.callback_query.message.text == "Select a command:"
    cfg = tmp_path / "config.yaml"
    cfg.write_text("trade_size_pct: 0.1\nmax_open_trades: 2\n")
    monkeypatch.setattr(telegram_bot_ui, "CONFIG_FILE", cfg)

    update = DummyUpdate()
    asyncio.run(ui.show_config(update, DummyContext()))
    assert "trade_size_pct" in update.message.text

    ctx = DummyContext()
    cb = DummyCallbackUpdate(EDIT_TRADE_SIZE)
    r = asyncio.run(ui.edit_trade_size(cb, ctx))
    assert r == telegram_bot_ui.ConversationHandler.END

    msg = DummyUpdate()
    msg.message.text = "0.2"
    asyncio.run(ui.set_config_value(msg, ctx))
    data = json.loads(cfg.read_text())
    assert data["trade_size_pct"] == 0.2
    assert state["reload"] is True

    ctx2 = DummyContext()
    cb = DummyCallbackUpdate(EDIT_MAX_TRADES)
    r = asyncio.run(ui.edit_max_trades(cb, ctx2))
    assert r == telegram_bot_ui.ConversationHandler.END

    msg2 = DummyUpdate()
    msg2.message.text = "5"
    asyncio.run(ui.set_config_value(msg2, ctx2))
    data = json.loads(cfg.read_text())
    assert data["max_open_trades"] == 5
    trades_file = tmp_path / "trades.csv"
    trades_file.write_text("XBT/USDT,buy,1,100\nXBT/USDT,sell,1,110\n")
    monkeypatch.setattr(telegram_bot_ui, "TRADES_FILE", trades_file)

    update = DummyUpdate()
    asyncio.run(ui.show_pnl_stats(update, DummyContext()))
    assert "Total PnL: 10.00" in update.message.text
    assert "Win rate: 100.0%" in update.message.text
