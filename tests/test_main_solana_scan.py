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
    async def fake_get(conf, timeout=None):
        calls.append((conf, timeout))
        return ["AAA/USDC"]
    monkeypatch.setattr(main, "get_solana_new_tokens", fake_get)
    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "recent_solana_tokens", deque())
    monkeypatch.setattr(main, "recent_solana_set", set())
    monkeypatch.setattr(main, "NEW_SOLANA_TOKENS", set())
    async def fake_fetch(sym):
        import pandas as pd

        return pd.DataFrame({"timestamp": [0], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]})

    async def fake_classify(sym, tf, df):
        return ("volatile", {})

    monkeypatch.setattr(main, "fetch_ohlcv_for_token", fake_fetch)
    monkeypatch.setattr(main, "classify_regime_cached", fake_classify)
    t = [100.0]
    monkeypatch.setattr(main.time, "time", lambda: t[0])
    last = await main.maybe_scan_solana_tokens(cfg, 0.0)
    assert list(main.symbol_priority_queue) == ["AAA/USDC"]
    assert main.NEW_SOLANA_TOKENS == {"AAA/USDC"}
    assert calls == [cfg["solana_scanner"]]
    monkeypatch.setattr(main, "SOLANA_SCAN_TASK", None, raising=False)
    t = [100.0]
    monkeypatch.setattr(main.time, "time", lambda: t[0])
    last = await main.maybe_scan_solana_tokens(cfg, 0.0)
    await main.SOLANA_SCAN_TASK
    assert list(main.symbol_priority_queue) == ["AAA/USDC", "AAA/USDC"]
    assert calls == [(cfg["solana_scanner"], 30.0)]
    t[0] += 30
    last2 = await main.maybe_scan_solana_tokens(cfg, last)
    assert last2 == last
    assert calls == [(cfg["solana_scanner"], 30.0)]
    t[0] += 60
    await main.maybe_scan_solana_tokens(cfg, last2)
    await main.SOLANA_SCAN_TASK
    assert calls == [(cfg["solana_scanner"], 30.0), (cfg["solana_scanner"], 30.0)]
