import asyncio
import logging
import pytest

class StopLoop(Exception):
    pass

class DummyNotifier:
    def __init__(self):
        self.token = "t"
        self.chat_id = "c"
        self.enabled = True
    def notify(self, text):
        pass

class DummyExchange:
    def fetch_balance(self):
        return {"USDT": {"free": 0}}


def test_init_logs_credentials_and_exchange(monkeypatch, caplog):
    import sys, types
    sys.modules.setdefault("redis", types.SimpleNamespace())
    sys.modules.setdefault("boto3", types.SimpleNamespace())
    sys.modules.setdefault("hvac", types.SimpleNamespace())
    sys.modules.setdefault("gspread", types.SimpleNamespace())
    sys.modules.setdefault(
        "oauth2client.service_account",
        types.SimpleNamespace(ServiceAccountCredentials=lambda *a, **k: None),
    )

    import crypto_bot.main as main

    caplog.set_level(logging.INFO)
    cfg = {"exchange": "kraken", "use_websocket": True}

    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "cooldown_configure", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "market_loader_configure", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "dotenv_values", lambda path: {})
    monkeypatch.setattr(main, "load_or_create", lambda: {"exchange": "kraken"})
    monkeypatch.setattr(main, "send_test_message", lambda *a, **k: True)
    monkeypatch.setattr(main.TelegramNotifier, "from_config", lambda cfg: DummyNotifier())
    monkeypatch.setattr(main, "log_balance", lambda *a, **k: None)
    monkeypatch.setattr(main, "RiskConfig", lambda *_a, **_k: None)

    class DummyRM:
        def __init__(self, *_a, **_k):
            raise StopLoop
    monkeypatch.setattr(main, "RiskManager", DummyRM)

    monkeypatch.setattr(main, "get_exchange", lambda cfg: (DummyExchange(), None))

    with pytest.raises(StopLoop):
        asyncio.run(main._main_impl())

    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "User credentials loaded" in text
    assert "Using kraken exchange" in text
    assert "websocket=True" in text
