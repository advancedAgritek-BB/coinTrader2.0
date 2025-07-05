from crypto_bot.utils.telegram_notifier import TelegramNotifier


def test_notify_uses_send_message(monkeypatch):
    calls = {}

    def fake_send(token, chat_id, text):
        calls['token'] = token
        calls['chat_id'] = chat_id
        calls['text'] = text
        return 'err'

    monkeypatch.setattr('crypto_bot.utils.telegram_notifier.send_message', fake_send)

    notifier = TelegramNotifier('t', 'c')
    err = notifier.notify('msg')

    assert err == 'err'
    assert calls == {'token': 't', 'chat_id': 'c', 'text': 'msg'}

