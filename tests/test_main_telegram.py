import asyncio
import pytest

import crypto_bot.main as main

class StopLoop(Exception):
    pass


def test_main_sends_telegram_check(monkeypatch):
    calls = {}

    def fake_send_test(token, chat_id, text="Bot started"):
        calls["args"] = (token, chat_id, text)
        return True

    monkeypatch.setattr(main, "send_test_message", fake_send_test)
    monkeypatch.setattr(main, "load_config", lambda: {})
    monkeypatch.setattr(main, "cooldown_configure", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "dotenv_values", lambda path: {})
    monkeypatch.setattr(main, "load_or_create", lambda: {"telegram_token": "t", "telegram_chat_id": "c"})
    monkeypatch.setattr(main, "get_exchange", lambda config: _raise_stop())

    asyncio.run(main.main())

    assert calls["args"] == ("t", "c", "Bot started")


def _raise_stop():
    raise StopLoop


def test_trade_updates_disabled(monkeypatch):
    calls = {"entry": 0, "exit": 0}

    monkeypatch.setattr(main, "report_entry", lambda *a, **k: calls.__setitem__("entry", calls["entry"] + 1))
    monkeypatch.setattr(main, "report_exit", lambda *a, **k: calls.__setitem__("exit", calls["exit"] + 1))

    monkeypatch.setattr(main, "send_test_message", lambda *a, **k: True)
    monkeypatch.setattr(main, "load_config", lambda: {"telegram": {"trade_updates": False}})
    monkeypatch.setattr(main, "cooldown_configure", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "dotenv_values", lambda path: {})
    monkeypatch.setattr(main, "load_or_create", lambda: {"telegram_token": "t", "telegram_chat_id": "c"})
    monkeypatch.setattr(main, "get_exchange", lambda config: _raise_stop())

    asyncio.run(main.main())

    assert calls["entry"] == 0
    assert calls["exit"] == 0


def test_status_updates_disabled(monkeypatch):
    notes = {}

    class DummyNotifier:
        def __init__(self):
            self.token = "t"
            self.chat_id = "c"
            self.enabled = True
            self.sent = []

        def notify(self, text):
            self.sent.append(text)

    async def fake_update(*a, **kw):
        notes["notifier"] = kw.get("notifier")
        raise StopLoop

    monkeypatch.setattr(main, "update_multi_tf_ohlcv_cache", fake_update)
    monkeypatch.setattr(main, "update_regime_tf_cache", fake_update)
    monkeypatch.setattr(main, "send_test_message", lambda *a, **k: True)
    monkeypatch.setattr(main, "load_config", lambda: {"telegram": {"status_updates": False}})
    monkeypatch.setattr(main, "cooldown_configure", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "dotenv_values", lambda path: {})
    monkeypatch.setattr(main, "load_or_create", lambda: {"telegram_token": "t", "telegram_chat_id": "c"})

    class DummyExchange:
        def fetch_balance(self):
            return {"USDT": {"free": 0}}

    monkeypatch.setattr(main, "get_exchange", lambda config: (DummyExchange(), None))
    monkeypatch.setattr(main.TelegramNotifier, "from_config", lambda cfg: DummyNotifier())
    monkeypatch.setattr(main, "RiskConfig", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "RiskManager", lambda *_a, **_k: (_ for _ in ()).throw(StopLoop))

    with pytest.raises(StopLoop):
        asyncio.run(main.main())

    assert notes.get("notifier") is None


def test_balance_change_notification(monkeypatch):
    msgs = []

    class DummyNotifier:
        def __init__(self):
            self.token = "t"
            self.chat_id = "c"
            self.enabled = True

        def notify(self, text):
            msgs.append(text)

    main.notify_balance_change(DummyNotifier(), 100.0, 120.0, True)

    assert msgs == ["Balance changed: 120.00 USDT"]

    
