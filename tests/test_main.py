import asyncio
import pytest
import crypto_bot.main as main

class StopLoop(Exception):
    pass

class DummyNotifier:
    def __init__(self):
        self.token = "t"
        self.chat_id = "c"
        self.enabled = True
        self.sent = []

    def notify(self, text):
        self.sent.append(text)

class FailingExchange:
    def fetch_balance(self):
        raise RuntimeError("boom")


def test_dry_run_continues_on_balance_error(monkeypatch):
    cfg = {"execution_mode": "dry_run"}
    notifier = DummyNotifier()

    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "cooldown_configure", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "dotenv_values", lambda path: {})
    monkeypatch.setattr(main, "load_or_create", lambda: {})
    monkeypatch.setattr(main, "send_test_message", lambda *_a, **_k: True)
    monkeypatch.setattr(main, "log_balance", lambda *_a, **_k: None)
    monkeypatch.setattr(main.TelegramNotifier, "from_config", lambda cfg: notifier)
    monkeypatch.setattr(main, "RiskConfig", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "RiskManager", lambda *_a, **_k: (_ for _ in ()).throw(StopLoop))
    monkeypatch.setattr(main, "get_exchange", lambda cfg: (FailingExchange(), None))

    with pytest.raises(StopLoop):
        asyncio.run(main._main_impl())

    assert any("API error" in msg for msg in notifier.sent)
