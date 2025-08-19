import asyncio
from collections import deque
import pandas as pd
import types, sys

from crypto_bot.phase_runner import BotContext
import crypto_bot.main as main

dummy = types.ModuleType("dummy")
for mod in ["telegram", "gspread", "scipy", "scipy.stats", "redis"]:
    sys.modules.setdefault(mod, dummy)

oauth_module = types.ModuleType("oauth2client")
oauth_module.service_account = types.SimpleNamespace(ServiceAccountCredentials=object)
sys.modules.setdefault("oauth2client", oauth_module)
sys.modules.setdefault("oauth2client.service_account", oauth_module.service_account)

def _ctx():
    df = pd.DataFrame({"high": [1, 2], "low": [0, 1], "close": [1, 2]})
    ctx = BotContext(
        positions={},
        df_cache={"1h": {"SOL/USDC": df}},
        regime_cache={},
        config={
            "timeframe": "1h",
            "symbols": ["SOL/USDC"],
            "symbol_batch_size": 5,
            "symbol": "ETH/USDT",
        },
    )
    ctx.exchange = object()
    return ctx

def test_single_symbol_no_duplicates(monkeypatch):
    ctx = _ctx()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("SOL/USDC", 1.0)], []

    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "calc_atr", lambda df, window=14: pd.Series([0.01]))
    monkeypatch.setattr(main, "compute_average_atr", lambda *_a, **_k: 0.01)
    monkeypatch.setattr(main, "symbol_priority_queue", deque())

    asyncio.run(main.fetch_candidates(ctx))

    assert set(ctx.current_batch) == {"ETH/USDT", "SOL/USDC"}


def test_disable_benchmark_symbols(monkeypatch):
    ctx = _ctx()
    ctx.config["benchmark_symbols"] = []

    async def fake_get_filtered_symbols(ex, cfg):
        return [("SOL/USDC", 1.0)], []

    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "calc_atr", lambda df, window=14: pd.Series([0.01]))
    monkeypatch.setattr(main, "compute_average_atr", lambda *_a, **_k: 0.01)
    monkeypatch.setattr(main, "symbol_priority_queue", deque())

    asyncio.run(main.fetch_candidates(ctx))

    assert set(ctx.current_batch) == {"SOL/USDC"}
