from crypto_bot.utils import trade_reporter
from crypto_bot.utils.telegram import TelegramNotifier


def test_report_entry_formats_and_sends(monkeypatch):
    calls = {}

    def fake_send(self, text):
        calls['self'] = self
        calls['text'] = text

    monkeypatch.setattr(TelegramNotifier, 'notify', fake_send)
    monkeypatch.setattr(trade_reporter.TelegramNotifier, 'notify', fake_send)
    monkeypatch.setattr('crypto_bot.utils.telegram.send_message', fake_send)

    notifier = TelegramNotifier('t', 'c')
    trade_reporter.report_entry(notifier, 'XBT/USDT', 'trend_bot', 0.876, 'long')

    assert isinstance(calls['self'], TelegramNotifier)
    assert calls['text'] == 'Entering LONG on XBT/USDT using trend_bot. Score: 0.88'


def test_report_exit_formats_and_sends(monkeypatch):
    calls = {}

    def fake_send(self, text):
        calls['self'] = self
        calls['text'] = text

    monkeypatch.setattr(TelegramNotifier, 'notify', fake_send)
    monkeypatch.setattr(trade_reporter.TelegramNotifier, 'notify', fake_send)
    monkeypatch.setattr('crypto_bot.utils.telegram.send_message', fake_send)
 
    notifier = TelegramNotifier('t', 'c')
    trade_reporter.report_exit(notifier, 'XBT/USDT', 'trend_bot', 10.123, 'short')

    assert calls['text'] == 'Exiting SHORT on XBT/USDT from trend_bot. PnL: 10.12'


def test_reporter_disabled(monkeypatch):
    calls = {"count": 0}

    def fake_send(token, chat_id, text):
        calls["count"] += 1

    monkeypatch.setattr("crypto_bot.utils.telegram.send_message", fake_send)

    notifier = TelegramNotifier(False, "t", "c")
    trade_reporter.report_entry(notifier, "XBT/USDT", "s", 0.0, "long")
    trade_reporter.report_exit(notifier, "XBT/USDT", "s", 0.0, "long")

    assert calls["count"] == 0
