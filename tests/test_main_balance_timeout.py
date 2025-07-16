import asyncio
import logging
import crypto_bot.main as main

class SlowExchange:
    async def fetch_balance(self):
        await asyncio.sleep(0.05)
        return {"USDT": {"free": 0}}

def test_main_aborts_on_balance_timeout(monkeypatch, caplog):
    caplog.set_level(logging.ERROR)
    cfg = {"http_timeout": 0.01}

    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "cooldown_configure", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "dotenv_values", lambda path: {})
    monkeypatch.setattr(main, "load_or_create", lambda: {})
    monkeypatch.setattr(main, "send_test_message", lambda *_a, **_k: True)
    monkeypatch.setattr(main, "log_balance", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "RiskConfig", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "RiskManager", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "get_exchange", lambda cfg: (SlowExchange(), None))

    asyncio.run(main.main())

    assert any("timed out" in r.getMessage() for r in caplog.records)
