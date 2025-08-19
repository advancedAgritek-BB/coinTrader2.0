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
        df_cache={"1h": {"BTC/USD": df}},
        regime_cache={},
        config={
            "timeframe": "1h",
            "symbols": ["BTC/USD"],
            "symbol_batch_size": 1,
            "solana_scanner": {"enabled": True},
        },
    )
    ctx.exchange = object()
    return ctx


def test_deduped_solana_queue(monkeypatch):
    ctx = _ctx()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("BTC/USD", 1.0)], []

    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "get_market_regime", lambda *_a, **_k: "volatile")
    monkeypatch.setattr(main, "compute_average_atr", lambda *_a, **_k: 0.01)
    monkeypatch.setattr(main, "calc_atr", lambda df, window=14: pd.Series([0.01]))
    monkeypatch.setattr(main, "get_solana_new_tokens", lambda cfg: ["SOL/USDC"])

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "recent_solana_tokens", deque(["SOL/USDC"]))
    monkeypatch.setattr(main, "recent_solana_set", {"SOL/USDC"})

    asyncio.run(main.fetch_candidates(ctx))

    assert ctx.current_batch == ["BTC/USD"]

