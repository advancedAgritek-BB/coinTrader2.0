import asyncio
import importlib.util
import types
from pathlib import Path
import sys
import pytest

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
