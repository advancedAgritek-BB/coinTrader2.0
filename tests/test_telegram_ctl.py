import time
import telegram_ctl as ctl


def test_set_get_page():
    ctl.callback_state.clear()
    ctl.set_page(1, "logs", 2)
    assert ctl.get_page(1, "logs") == 2
    assert "1" in ctl.callback_state


def test_expiration(monkeypatch):
    ctl.callback_state.clear()
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    ctl.set_page(1, "logs", 1)
    assert ctl.get_page(1, "logs") == 1

    monkeypatch.setattr(time, "time", lambda: now + ctl.callback_timeout + 1)
    assert ctl.get_page(1, "logs") == 0
    assert ctl.callback_state == {}
import asyncio
import types
import pytest

try:
    import crypto_bot.telegram_ctl as telegram_ctl
except Exception as e:  # pragma: no cover - module may not exist
    telegram_ctl = None


class DummyUpdate:
    def __init__(self, user_id=1):
        self.message = types.SimpleNamespace(text=None)
        self.effective_user = types.SimpleNamespace(id=user_id)

    async def reply_text(self, text):
        self.message.text = text


class DummyContext:
    pass


class DummyBotController:
    def __init__(self):
        self.calls = []
        self.state = {}

    async def start(self):
        self.calls.append("start")

    async def stop(self):
        self.calls.append("stop")

    async def status(self):
        self.calls.append("status")

    async def log(self):
        self.calls.append("log")

    async def rotate_now(self):
        self.calls.append("rotate_now")

    async def toggle_mode(self):
        self.calls.append("toggle_mode")

    async def signals(self):
        self.calls.append("signals")

    async def balance(self):
        self.calls.append("balance")

    async def trades(self):
        self.calls.append("trades")

    async def reload(self):
        self.calls.append("reload")
        self.state["reload"] = True
    async def reload_config(self):
        self.calls.append("reload_config")
        self.state["reload"] = True


@pytest.mark.skipif(telegram_ctl is None, reason="telegram_ctl module missing")
class TestTelegramCtl:
    def setup_method(self):
        self.controller = DummyBotController()
        self.tg = telegram_ctl.TelegramCtl(self.controller, admin_id=1)

    @pytest.mark.asyncio
    async def test_admin_check(self):
        update = DummyUpdate(user_id=2)
        await self.tg.start_cmd(update, DummyContext())
        assert not self.controller.calls
        assert update.message.text == "Unauthorized"

    @pytest.mark.asyncio
    async def test_commands_call_controller(self):
        update = DummyUpdate()
        await self.tg.start_cmd(update, DummyContext())
        await self.tg.stop_cmd(update, DummyContext())
        await self.tg.status_cmd(update, DummyContext())
        await self.tg.log_cmd(update, DummyContext())
        await self.tg.rotate_now_cmd(update, DummyContext())
        await self.tg.toggle_mode_cmd(update, DummyContext())
        await self.tg.signals_cmd(update, DummyContext())
        await self.tg.balance_cmd(update, DummyContext())
        await self.tg.trades_cmd(update, DummyContext())
        await self.tg.reload_cmd(update, DummyContext())
        assert self.controller.calls == [
            "start",
            "stop",
            "status",
            "log",
            "rotate_now",
            "toggle_mode",
            "signals",
            "balance",
            "trades",
            "reload_config",
        ]

    @pytest.mark.asyncio
    async def test_pagination(self):
        update = DummyUpdate()
        long_text = "line\n" * 50
        pages = telegram_ctl._paginate(long_text)
        assert len(pages) > 1
        await self.tg._send_pages(update, pages)
        assert update.message.text is not None

    @pytest.mark.asyncio
    async def test_heartbeat_start_stop(self):
        hb = self.tg.start_heartbeat()
        assert isinstance(hb, asyncio.Task)
        self.tg.stop_heartbeat()
        await asyncio.sleep(0)
        await self.tg.stop_heartbeat()
        assert hb.cancelled() or hb.done()

    @pytest.mark.asyncio
    async def test_reload_cmd_sets_flag(self, monkeypatch):
        import sys, types
        stub = types.ModuleType("crypto_bot.main")
        load_calls = []

        def fake_load():
            load_calls.append(True)
            return {}

        def maybe_reload_config(state, config):
            if state.get("reload"):
                cfg = fake_load()
                config.clear()
                config.update(cfg)
                state.pop("reload", None)

        stub.load_config = fake_load
        stub.maybe_reload_config = maybe_reload_config
        monkeypatch.setitem(sys.modules, "crypto_bot.main", stub)
        main = stub

        update = DummyUpdate()
        await self.tg.reload_cmd(update, DummyContext())
        assert "reload_config" in self.controller.calls

        config = {}
        main.maybe_reload_config(self.controller.state, config)
        assert not self.controller.state.get("reload")
        assert load_calls
from crypto_bot.telegram_ctl import status_loop
from crypto_bot.utils.telegram import TelegramNotifier


class DummyController:
    def get_status(self):
        return "running"

    def list_positions(self):
        return ["XBT/USDT long"]


class DummyNotifier(TelegramNotifier):
    def __init__(self):
        super().__init__(True, "t", "c")
        self.sent = []

    def notify(self, text):
        self.sent.append(text)


class StopLoop(Exception):
    pass


def test_status_loop_sends_summary(monkeypatch):
    controller = DummyController()
    notifier = DummyNotifier()

    async def fake_sleep(_):
        raise StopLoop

    import crypto_bot.telegram_ctl as telegram_ctl
    monkeypatch.setattr(telegram_ctl.asyncio, "sleep", fake_sleep)

    with pytest.raises(StopLoop):
        asyncio.run(status_loop(controller, [notifier], update_interval=0))

    assert notifier.sent
    assert "running" in notifier.sent[0]
    assert "XBT/USDT" in notifier.sent[0]
