import asyncio
from collections import deque
import pandas as pd

import types
import sys
from crypto_bot.phase_runner import BotContext

dummy = types.ModuleType("crypto_bot.utils.regime_pnl_tracker")
dummy.log_trade = lambda *a, **k: None
dummy.get_recent_win_rate = lambda *a, **k: 1.0
sys.modules["crypto_bot.utils.regime_pnl_tracker"] = dummy

import crypto_bot.main as main


def test_fetch_candidates_adapts(monkeypatch):
    df = pd.DataFrame({"high": [1, 2], "low": [0, 1], "close": [1, 2]})
    ctx = BotContext(
        positions={},
        df_cache={"1h": {"BTC/USD": df}},
        regime_cache={},
        config={
            "timeframe": "1h",
            "symbols": ["BTC/USD", "ETH/USD"],
            "symbol_batch_size": 1,
            "adaptive_scan": {"enabled": True, "atr_baseline": 0.01, "max_factor": 2.0},
        },
    )
    ctx.exchange = object()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("BTC/USD", 1.0), ("ETH/USD", 0.9)]

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "calc_atr", lambda df, window=14: 0.02)

    asyncio.run(main.fetch_candidates(ctx))

    assert ctx.volatility_factor == 2.0
    assert len(ctx.current_batch) == 2

