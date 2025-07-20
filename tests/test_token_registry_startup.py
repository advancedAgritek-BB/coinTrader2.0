import asyncio
import importlib
import pytest

import crypto_bot.main as main

class StopLoop(Exception):
    pass


def _raise_stop(*_a, **_k):
    raise StopLoop


def test_load_token_mints_once(monkeypatch):
    cfg = {}
    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "cooldown_configure", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "dotenv_values", lambda path: {})
    monkeypatch.setattr(main, "load_or_create", lambda: {})
    monkeypatch.setattr(main, "send_test_message", lambda *_a, **_k: True)
    monkeypatch.setattr(main, "log_balance", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "RiskConfig", lambda *_a, **_k: None)
    monkeypatch.setattr(main, "RiskManager", lambda *_a, **_k: None)
    monkeypatch.setattr(main.asyncio, "sleep", lambda *_a: None)

    registry = importlib.import_module("crypto_bot.utils.token_registry")
    registry.TOKEN_MINTS.clear()
    registry._LOADED = False
    calls = {"load": 0}

    async def fake_load_token_mints():
        if registry._LOADED:
            return {}
        calls["load"] += 1
        registry._LOADED = True
        return {"AAA": "mint"}

    monkeypatch.setattr(registry, "load_token_mints", fake_load_token_mints)
    monkeypatch.setattr(main, "get_exchange", lambda cfg: (_raise_stop()))

    with pytest.raises(StopLoop):
        asyncio.run(main._main_impl())

    assert calls["load"] == 1
    assert registry.TOKEN_MINTS["AAA"] == "mint"

    with pytest.raises(StopLoop):
        asyncio.run(main._main_impl())

    assert calls["load"] == 1
