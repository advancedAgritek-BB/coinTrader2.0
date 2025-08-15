import asyncio
from datetime import datetime
import importlib.util
import pathlib
import sys
import types
import aiohttp
import pytest

# Create minimal package structure to load token_registry in isolation
pkg_root = types.ModuleType("crypto_bot")
pkg_root.__path__ = [str(pathlib.Path("crypto_bot"))]
utils_pkg = types.ModuleType("crypto_bot.utils")
utils_pkg.__path__ = [str(pathlib.Path("crypto_bot/utils"))]
pkg_root.utils = utils_pkg
sys.modules.setdefault("crypto_bot", pkg_root)
sys.modules.setdefault("crypto_bot.utils", utils_pkg)

gecko_mod = types.ModuleType("crypto_bot.utils.gecko")
gecko_mod.gecko_request = lambda url: {}
sys.modules.setdefault("crypto_bot.utils.gecko", gecko_mod)

spec = importlib.util.spec_from_file_location(
    "crypto_bot.utils.token_registry",
    pathlib.Path(__file__).resolve().parents[1] / "crypto_bot" / "utils" / "token_registry.py",
)
token_registry = importlib.util.module_from_spec(spec)
sys.modules["crypto_bot.utils.token_registry"] = token_registry
spec.loader.exec_module(token_registry)

from tests.dummy_aiohttp import DummyResp, DummySession as BaseDummySession


class DummySession(BaseDummySession):
    def __init__(self, pump_data):
        super().__init__()
        self.pump_data = pump_data
        self.calls = {"first": True}

    async def get(self, url, timeout=10):
        if self.calls["first"]:
            self.calls["first"] = False
            req = types.SimpleNamespace(real_url=url)
            raise aiohttp.ClientResponseError(req, (), status=429, message="Too Many Requests")
        if "pump.fun" in url:
            return DummyResp(self.pump_data)
        return DummyResp([])


def test_monitor_new_tokens_filters_and_backoff(monkeypatch):
    now = datetime.utcnow().isoformat()
    pump_data = [
        {
            "symbol": "GOOD",
            "mint": "GOODMINT",
            "created_at": now,
            "initial_buy": True,
            "market_cap": 1,
            "twitter": "x",
        },
        {
            "symbol": "BAD1",
            "mint": "BAD1M",
            "created_at": now,
            "initial_buy": False,
            "market_cap": 1,
            "twitter": "x",
        },
        {
            "symbol": "BAD2",
            "mint": "BAD2M",
            "created_at": now,
            "initial_buy": True,
            "market_cap": 0,
            "twitter": "x",
        },
        {
            "symbol": "BAD3",
            "mint": "BAD3M",
            "created_at": now,
            "initial_buy": True,
            "market_cap": 1,
            "twitter": None,
        },
    ]

    session = DummySession(pump_data)
    aiohttp_mod = types.SimpleNamespace(ClientSession=lambda: session, ClientError=Exception)
    monkeypatch.setattr(token_registry, "aiohttp", aiohttp_mod)

    monkeypatch.setattr(token_registry, "TOKEN_MINTS", {})
    monkeypatch.setattr(token_registry, "_write_cache", lambda: None)

    async def fake_run_ml_trainer():
        pass

    monkeypatch.setattr(token_registry, "_run_ml_trainer", fake_run_ml_trainer)

    pkg = types.ModuleType("coinTrader_Trainer")
    sys.modules.setdefault("coinTrader_Trainer", pkg)
    trainer = types.ModuleType("coinTrader_Trainer.ml_trainer")
    async def fake_fetch():
        pass
    trainer.fetch_data_range_async = fake_fetch
    sys.modules["coinTrader_Trainer.ml_trainer"] = trainer

    calls = []
    orig_sleep = asyncio.sleep

    async def fake_sleep(delay):
        calls.append(delay)
        await orig_sleep(0)
        if len(calls) >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(token_registry.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(token_registry, "POLL_INTERVAL", 0)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(token_registry.monitor_new_tokens())

    assert calls and calls[0] == 1
    assert "GOOD" in token_registry.TOKEN_MINTS
    assert "BAD1" not in token_registry.TOKEN_MINTS
    assert "BAD2" not in token_registry.TOKEN_MINTS
    assert "BAD3" not in token_registry.TOKEN_MINTS
