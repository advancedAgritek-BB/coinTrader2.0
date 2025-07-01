import types
from crypto_bot.telegram_bot_ui import TelegramBotUI

class DummyUpdater:
    def __init__(self, *a, **k):
        self.dispatcher = types.SimpleNamespace(add_handler=lambda h: None)

    def start_polling(self):
        pass

    def stop(self):
        pass


class DummyMessage:
    def __init__(self):
        self.text = None

    def reply_text(self, text):
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
    ui = TelegramBotUI(
        "token",
        "chat",
        state,
        log_file,
        rotator=rotator,
        exchange=exchange,
        wallet="addr",
    )
    return ui, log_file


def test_start_stop_toggle(monkeypatch, tmp_path):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.Updater", DummyUpdater)
    state = {"running": False, "mode": "cex"}
    ui, _ = make_ui(tmp_path, state)

    update = DummyUpdate()
    ui.start_cmd(update, DummyContext())
    assert state["running"] is True
    assert update.message.text == "Trading started"

    ui.stop_cmd(update, DummyContext())
    assert state["running"] is False
    assert update.message.text == "Trading stopped"

    ui.toggle_mode_cmd(update, DummyContext())
    assert state["mode"] == "onchain"
    assert update.message.text == "Mode set to onchain"


def test_log_and_rotate(monkeypatch, tmp_path):
    monkeypatch.setattr("crypto_bot.telegram_bot_ui.Updater", DummyUpdater)
    rotator = DummyRotator()
    exchange = DummyExchange()
    state = {"running": True, "mode": "cex"}
    ui, log_file = make_ui(tmp_path, state, rotator=rotator, exchange=exchange)

    update = DummyUpdate()
    ui.log_cmd(update, DummyContext())
    assert "line2" in update.message.text

    ui.rotate_now_cmd(update, DummyContext())
    assert rotator.called is True
    assert update.message.text == "Portfolio rotated"

    ui.status_cmd(update, DummyContext())
    assert "Running: True" in update.message.text

