from collections import deque
import pandas as pd
import types, sys
import asyncio

from crypto_bot.phase_runner import BotContext
import crypto_bot.main as main

# provide dummy modules for optional dependencies
_dummy = types.ModuleType("dummy")
for mod in ["telegram", "gspread", "scipy", "scipy.stats", "redis"]:
    sys.modules.setdefault(mod, _dummy)

_oauth = types.ModuleType("oauth2client")
_oauth.service_account = types.SimpleNamespace(ServiceAccountCredentials=object)
sys.modules.setdefault("oauth2client", _oauth)
sys.modules.setdefault("oauth2client.service_account", _oauth.service_account)


def test_symbols_with_no_data_skipped(monkeypatch):
    ctx = BotContext(
        positions={},
        df_cache={},
        regime_cache={},
        config={"timeframe": "1h", "symbols": ["FOO/USDC"], "symbol_batch_size": 5},
    )
    ctx.exchange = object()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("FOO/USDC", 1.0)], []

    async def fake_update_multi(*args, **kwargs):
        return {"1h": {"FOO/USDC": pd.DataFrame()}}

    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "calc_atr", lambda df, period=14: 0.01)
    monkeypatch.setattr(main, "compute_average_atr", lambda *_a, **_k: 0.01)
    monkeypatch.setattr(main, "update_multi_tf_ohlcv_cache", fake_update_multi)

    async def fake_update_regime(*args, **kwargs):
        return {}

    monkeypatch.setattr(main, "update_regime_tf_cache", fake_update_regime)
    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "no_data_symbols", set())

    asyncio.run(main.fetch_candidates(ctx))
    assert "FOO/USDC" in ctx.current_batch

    asyncio.run(main.update_caches(ctx))
    assert "FOO/USDC" in main.no_data_symbols

    asyncio.run(main.fetch_candidates(ctx))
    assert "FOO/USDC" not in ctx.current_batch

