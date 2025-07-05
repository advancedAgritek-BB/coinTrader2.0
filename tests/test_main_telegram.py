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

    with pytest.raises(StopLoop):
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

    with pytest.raises(StopLoop):
        asyncio.run(main.main())

    assert calls["entry"] == 0
    assert calls["exit"] == 0

    
