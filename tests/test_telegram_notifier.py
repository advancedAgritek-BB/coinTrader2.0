import pytest
import types
import time
import importlib.util
from pathlib import Path
import sys

path = Path(__file__).resolve().parents[1] / "crypto_bot" / "utils" / "telegram.py"
spec = importlib.util.spec_from_file_location(
    "crypto_bot.utils.telegram", path, submodule_search_locations=[str(path.parent)]
)
telegram_pkg = types.ModuleType("crypto_bot.utils")
telegram_pkg.__path__ = [str(path.parent)]
sys.modules.setdefault("crypto_bot", types.ModuleType("crypto_bot"))
sys.modules["crypto_bot"].__path__ = [str(path.parent.parent)]
sys.modules["crypto_bot"].utils = telegram_pkg
sys.modules["crypto_bot.utils"] = telegram_pkg
telegram = importlib.util.module_from_spec(spec)
sys.modules["crypto_bot.utils.telegram"] = telegram
spec.loader.exec_module(telegram)
telegram_pkg.telegram = telegram
TelegramNotifier = telegram.TelegramNotifier


def test_notify_uses_send_message(monkeypatch):
    calls = {}

    def fake_send(token, chat_id, text):
        calls['token'] = token
        calls['chat_id'] = chat_id
        calls['text'] = text
        return 'err'

    monkeypatch.setattr('crypto_bot.utils.telegram.send_message', fake_send)

    notifier = TelegramNotifier(True, 't', 'c')
    err = notifier.notify('msg')

    assert err == 'err'
    assert calls == {'token': 't', 'chat_id': 'c', 'text': 'msg'}



def test_notify_calls_send_message_when_enabled(monkeypatch):
    calls = {}
    def fake_send(token, chat_id, text):
        calls['args'] = (token, chat_id, text)
        return 'ok'
    monkeypatch.setattr('crypto_bot.utils.telegram.send_message', fake_send)
    notifier = TelegramNotifier(True, 't', 'c')
    res = notifier.notify('msg')
    assert calls['args'] == ('t', 'c', 'msg')
    assert res == 'ok'


def test_notify_noop_when_disabled(monkeypatch):
    called = False
    def fake_send(*a, **k):
        nonlocal called
        called = True
    monkeypatch.setattr('crypto_bot.utils.telegram.send_message', fake_send)
    notifier = TelegramNotifier(False, 't', 'c')
    res = notifier.notify('msg')
    assert called is False
    assert res is None


def test_set_admin_ids_accepts_int(monkeypatch):
    import crypto_bot.utils.telegram as tg
    tg.set_admin_ids(123)
    assert tg.is_admin("123")
    tg.set_admin_ids([])


@pytest.mark.asyncio
async def test_notify_async_respects_message_interval(monkeypatch):
    import crypto_bot.utils.telegram as tg

    send_times = []

    def fake_send(token, chat_id, text):
        send_times.append(current["t"])
        return None

    monkeypatch.setattr(tg, "send_message", fake_send)

    current = {"t": 100.0}

    def fake_time():
        return current["t"]

    async def fake_sleep(sec):
        current["t"] += sec

    monkeypatch.setattr(time, "time", fake_time)
    monkeypatch.setattr(tg.asyncio, "sleep", fake_sleep)

    notifier = tg.TelegramNotifier(True, "t", "c", message_interval=1.0)

    await notifier.notify_async("a")
    current["t"] = 100.2
    await notifier.notify_async("b")

    assert len(send_times) == 2
    assert send_times[1] - send_times[0] >= 1.0


@pytest.mark.asyncio
async def test_notify_async_respects_max_messages_per_minute(monkeypatch):
    import crypto_bot.utils.telegram as tg
    send_times = []

    def fake_send(token, chat_id, text):
        send_times.append(current["t"])
        return None

    monkeypatch.setattr(tg, "send_message", fake_send)

    current = {"t": 200.0}

    def fake_time():
        return current["t"]

    async def fake_sleep(sec):
        current["t"] += sec

    monkeypatch.setattr(time, "time", fake_time)
    monkeypatch.setattr(tg.asyncio, "sleep", fake_sleep)

    notifier = tg.TelegramNotifier(True, "t", "c", message_interval=0.0, max_per_minute=2)

    await notifier.notify_async("1")
    current["t"] = 200.1
    await notifier.notify_async("2")
    current["t"] = 200.2
    await notifier.notify_async("3")

    assert len(send_times) == 3
    assert send_times[2] - send_times[0] >= 60


@pytest.mark.asyncio
async def test_notify_async_uses_send_message(monkeypatch):
    calls = {}

    def fake_send(token, chat_id, text):
        calls['token'] = token
        calls['chat_id'] = chat_id
        calls['text'] = text
        return 'err'

    monkeypatch.setattr('crypto_bot.utils.telegram.send_message', fake_send)

    notifier = TelegramNotifier(True, 't', 'c')
    err = await notifier.notify_async('msg')

    assert err == 'err'
    assert calls == {'token': 't', 'chat_id': 'c', 'text': 'msg'}


def test_notify_runs_notify_async(monkeypatch):
    calls = {}

    async def fake_async(self, text):
        calls['text'] = text
        return 'ok'

    monkeypatch.setattr(TelegramNotifier, 'notify_async', fake_async)

    notifier = TelegramNotifier(True, 't', 'c')
    res = notifier.notify('msg')

    assert res == 'ok'
    assert calls['text'] == 'msg'
