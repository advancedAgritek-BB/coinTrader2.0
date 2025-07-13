import asyncio
import builtins
import pytest

import crypto_bot.main as main

class StopLoop(Exception):
    pass


def test_main_dry_run_skips_fetch_balance(monkeypatch):
    calls = {"fetch": 0, "wallet": None}

    class DummyNotifier:
        def __init__(self):
            self.token = "t"
            self.chat_id = "c"
            self.enabled = True

        def notify(self, text):
            pass

    class DummyExchange:
        def __init__(self):
            self.load_markets = lambda: None

        def fetch_balance(self):
            calls["fetch"] += 1
            return {"USDT": {"free": 0}}

    class DummyWallet:
        def __init__(self, bal, *_a):
            calls["wallet"] = bal
            self.balance = bal

    monkeypatch.setattr(main, "send_test_message", lambda *a, **k: True)
    monkeypatch.setattr(main, "load_config", lambda: {"execution_mode": "dry_run"})
    monkeypatch.setattr(main, "cooldown_configure", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "dotenv_values", lambda path: {})
    monkeypatch.setattr(main, "load_or_create", lambda: {"telegram_token": "t", "telegram_chat_id": "c"})
    monkeypatch.setattr(main.TelegramNotifier, "from_config", lambda cfg: DummyNotifier())
    monkeypatch.setattr(main, "get_exchange", lambda config: (DummyExchange(), None))
    monkeypatch.setattr(main, "PaperWallet", DummyWallet)
    monkeypatch.setattr(main, "RiskConfig", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "RiskManager", lambda *_a, **_k: (_ for _ in ()).throw(StopLoop))
    monkeypatch.setattr(builtins, "input", lambda prompt="": "123")

    with pytest.raises(StopLoop):
        asyncio.run(main._main_impl())

    assert calls["fetch"] == 0
    assert calls["wallet"] == 123.0
