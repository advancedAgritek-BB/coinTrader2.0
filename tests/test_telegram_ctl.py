import asyncio
import types
import pytest

from crypto_bot.telegram_ctl import status_loop
from crypto_bot.utils.telegram import TelegramNotifier


class DummyController:
    def get_status(self):
        return "running"

    def list_positions(self):
        return ["BTC/USDT long"]


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
    assert "BTC/USDT" in notifier.sent[0]
