from crypto_bot.utils.telegram import TelegramNotifier


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
