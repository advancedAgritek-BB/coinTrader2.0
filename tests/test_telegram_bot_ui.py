import types
import asyncio
import json
import crypto_bot.telegram_bot_ui as telegram_bot_ui
from crypto_bot.telegram_bot_ui import TelegramBotUI
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


class DummyUpdate:
    def __init__(self):
        self.message = DummyMessage()


class DummyContext:
    pass


class DummyExchange:
    def fetch_balance(self):
        return {"BTC": {"total": 1}}


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
    monkeypatch.setattr(
        "crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder
    )
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
    monkeypatch.setattr(
        "crypto_bot.telegram_bot_ui.ApplicationBuilder", DummyBuilder
    )
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

    update = DummyUpdate()
    asyncio.run(ui.show_trades(update, DummyContext()))
    assert "BTC/USDT" in update.message.text

