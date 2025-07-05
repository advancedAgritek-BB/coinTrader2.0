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

    
