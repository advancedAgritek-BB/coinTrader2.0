import asyncio
import importlib
import pytest

import crypto_bot.main as main

class StopLoop(Exception):
    pass


def _raise_stop(*_a, **_k):
    raise StopLoop


def test_load_token_mints_once(monkeypatch, tmp_path):
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
    monkeypatch.setattr(registry, "CACHE_FILE", tmp_path / "token_mints.json", raising=False)
    calls = {"load": 0}

    async def fake_load_token_mints():
        if registry._LOADED:
            return {}
        calls["load"] += 1
        registry._LOADED = True
        return {"AAA": "mint"}

    monkeypatch.setattr(registry, "load_token_mints", fake_load_token_mints)
    monkeypatch.setattr(main, "get_exchanges", lambda cfg: (_raise_stop()))

    with pytest.raises(StopLoop):
        asyncio.run(main._main_impl())

    assert calls["load"] == 1
    assert registry.TOKEN_MINTS["AAA"] == "mint"

    with pytest.raises(StopLoop):
        asyncio.run(main._main_impl())

    assert calls["load"] == 1


def test_registry_update_loop_refreshes(monkeypatch):
    calls = {"load": 0, "sleep": 0}

    async def fake_load(force_refresh=False):
        calls["load"] += 1
        return {}

    import types, sys, importlib
    registry = types.ModuleType("crypto_bot.utils.token_registry")
    registry.load_token_mints = fake_load
    registry.set_token_mints = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "crypto_bot.utils.token_registry", registry)
    monkeypatch.setattr(importlib.import_module("crypto_bot.utils"), "token_registry", registry)

    async def fake_sleep(_):
        calls["sleep"] += 1
        if calls["sleep"] >= 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(main.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(main.registry_update_loop(0))

    assert calls["load"] > 1
