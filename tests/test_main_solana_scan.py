import types
import sys

dummy = types.ModuleType("dummy")
dummy.AsyncClient = object
sys.modules.setdefault("oauth2client", dummy); sys.modules.setdefault("oauth2client.service_account", dummy)
dummy.ServiceAccountCredentials = object
for mod in ["telegram", "gspread", "scipy", "scipy.stats", "redis"]:
    sys.modules.setdefault(mod, dummy)
for mod in ["solana", "solana.rpc", "solana.rpc.async_api"]:
    sys.modules.setdefault(mod, dummy)

import asyncio
from collections import deque
import pytest

import crypto_bot.main as main

@pytest.mark.asyncio
async def test_maybe_scan_solana_tokens(monkeypatch):
    cfg = {"solana_scanner": {"enabled": True, "interval_minutes": 1}}
    calls = []
    async def fake_get(conf):
        calls.append(conf)
        return ["AAA/USDC"]
    monkeypatch.setattr(main, "get_solana_new_tokens", fake_get)
    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "recent_solana_tokens", deque())
    monkeypatch.setattr(main, "recent_solana_set", set())
    t = [100.0]
    monkeypatch.setattr(main.time, "time", lambda: t[0])
    last = await main.maybe_scan_solana_tokens(cfg, 0.0)
    assert list(main.symbol_priority_queue) == ["AAA/USDC", "AAA/USDC"]
    assert calls == [cfg["solana_scanner"]]
    t[0] += 30
    last2 = await main.maybe_scan_solana_tokens(cfg, last)
    assert last2 == last
    assert calls == [cfg["solana_scanner"]]
    t[0] += 60
    await main.maybe_scan_solana_tokens(cfg, last2)
    assert calls == [cfg["solana_scanner"], cfg["solana_scanner"]]
