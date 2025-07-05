from crypto_bot.utils import trade_reporter
from crypto_bot.utils.telegram import TelegramNotifier


def test_report_entry_formats_and_sends(monkeypatch):
    calls = {}

    def fake_send(token, chat_id, text):
        calls['token'] = token
        calls['chat_id'] = chat_id
        calls['text'] = text

    monkeypatch.setattr('crypto_bot.utils.telegram.send_message', fake_send)

    notifier = TelegramNotifier('t', 'c')
    trade_reporter.report_entry(notifier, 'BTC/USDT', 'trend_bot', 0.876, 'long')

    assert calls['token'] == 't'
    assert calls['chat_id'] == 'c'
    assert calls['text'] == 'Entering LONG on BTC/USDT using trend_bot. Score: 0.88'


def test_report_exit_formats_and_sends(monkeypatch):
    calls = {}

    def fake_send(token, chat_id, text):
        calls['token'] = token
        calls['chat_id'] = chat_id
        calls['text'] = text

    monkeypatch.setattr('crypto_bot.utils.telegram.send_message', fake_send)

    notifier = TelegramNotifier('t', 'c')
    trade_reporter.report_exit(notifier, 'BTC/USDT', 'trend_bot', 10.123, 'short')

    assert calls['text'] == 'Exiting SHORT on BTC/USDT from trend_bot. PnL: 10.12'
