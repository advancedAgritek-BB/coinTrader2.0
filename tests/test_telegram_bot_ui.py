import types
import asyncio
import json
import crypto_bot.telegram_bot_ui as telegram_bot_ui
from crypto_bot.telegram_bot_ui import TelegramBotUI
from crypto_bot.telegram_bot_ui import (
    TelegramBotUI,
    SIGNALS,
    BALANCE,
    TRADES,
)
from crypto_bot.utils.telegram import TelegramNotifier


class DummyApplication:
    def __init__(self, *a, **k):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)

    def run_polling(self, *a, **k):
        pass

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

    async def reply_text(self, text):
        self.text = text

    async def edit_text(self, text, reply_markup=None):
        self.text = text


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
    pass


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


def make_ui(tmp_path, state, rotator=None, exchange=None):
    log_file = tmp_path / "bot.log"
    log_file.write_text("line1\nline2\n")
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


def test_start_stop_toggle(monkeypatch, tmp_path):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    state = {"running": False, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state)

    update = DummyUpdate()
    asyncio.run(ui.start_cmd(update, DummyContext()))
    assert state["running"] is True
    assert update.message.text == "Trading started"

    asyncio.run(ui.stop_cmd(update, DummyContext()))
    assert state["running"] is False
    assert update.message.text == "Trading stopped"

    asyncio.run(ui.toggle_mode_cmd(update, DummyContext()))
    assert state["mode"] == "onchain"
    assert update.message.text == "Mode set to onchain"


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


def test_menu_signals_balance_trades(monkeypatch, tmp_path):
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
    trades_file.write_text("BTC/USDT,buy,1,100\n")
    monkeypatch.setattr(telegram_bot_ui, "TRADES_FILE", trades_file)

    update = DummyUpdate()
    asyncio.run(ui.menu_cmd(update, DummyContext()))
    assert "rotate_now" in update.message.text

    update = DummyUpdate()
    asyncio.run(ui.show_signals(update, DummyContext()))
    assert "BTC" in update.message.text

    update = DummyUpdate()
    asyncio.run(ui.show_balance(update, DummyContext()))
    assert "BTC" in update.message.text



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

    update = DummyUpdate()
    asyncio.run(ui.show_trades(update, DummyContext()))
    assert "BTC/USDT" in update.message.text

def test_menu_callbacks(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "crypto_bot.telegram_bot_ui.ApplicationBuilder",
        DummyBuilder,
    )
    asset_file = tmp_path / "scores.json"
    asset_file.write_text('{"BTC/USDT": 0.5}')
    trades_file = tmp_path / "trades.csv"
    trades_file.write_text("BTC/USDT,buy,1,100,t1\n")

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
    assert "BTC/USDT" in update.callback_query.message.text

    update = DummyCallbackUpdate()
    update.callback_query.data = BALANCE
    asyncio.run(ui.show_balance(update, DummyContext()))
    assert "Free USDT" in update.callback_query.message.text

    update = DummyCallbackUpdate()
    update.callback_query.data = TRADES
    asyncio.run(ui.show_trades(update, DummyContext()))
    assert "+5.00" in update.callback_query.message.text


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

    update = DummyUpdate()
    asyncio.run(ui.show_balance(update, DummyContext()))
    assert "BTC" in update.message.text


def test_command_cooldown(monkeypatch, tmp_path):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder)
    t = {"now": 0}
    monkeypatch.setattr(telegram_bot_ui.time, "time", lambda: t["now"])
    state = {"running": False, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state)
    ui.command_cooldown = 5

    update1 = DummyUpdate()
    asyncio.run(ui.start_cmd(update1, DummyContext()))
    assert update1.message.text == "Trading started"

    t["now"] = 2
    update2 = DummyUpdate()
    asyncio.run(ui.start_cmd(update2, DummyContext()))
    assert update2.message.text == "Please wait"

    t["now"] = 6
    update3 = DummyUpdate()
    asyncio.run(ui.start_cmd(update3, DummyContext()))
    assert update3.message.text == "Trading started"
