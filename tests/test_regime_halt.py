import asyncio
import logging
from collections import deque
import pandas as pd
import types, sys

from crypto_bot.phase_runner import BotContext
import crypto_bot.main as main

# stub modules similar to other tests
_dummy = types.ModuleType("dummy")
for mod in ["telegram", "gspread", "scipy", "scipy.stats", "redis"]:
    sys.modules.setdefault(mod, _dummy)
for mod in ["solana", "solana.rpc", "solana.rpc.async_api"]:
    sys.modules.setdefault(mod, _dummy)
_dummy.AsyncClient = object
sys.modules.setdefault("keyring", _dummy)


def test_trading_disabled_flag(monkeypatch, caplog):
    df = pd.DataFrame({"high": [1, 2], "low": [0, 1], "close": [1, 2]})
    ctx = BotContext(
        positions={},
        df_cache={"1h": {"BTC/USD": df}},
        regime_cache={},
        config={
            "timeframe": "1h",
            "symbols": ["BTC/USD"],
            "symbol_batch_size": 1,
            "benchmark_symbols": [],
        },
    )
    ctx.exchange = object()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("BTC/USD", 1.0)], []

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "calc_atr", lambda df, period=14: pd.Series([0.02]))

    async def fake_regime(*_a, **_k):
        return "risk_off"

    monkeypatch.setattr(main, "get_market_regime", fake_regime)
    monkeypatch.setattr(main, "compute_average_atr", lambda *_a, **_k: 0.01)

    caplog.set_level(logging.WARNING)

    asyncio.run(main.fetch_candidates(ctx))
    assert ctx.trading_disabled
    assert "Trading disabled by regime model: risk_off" in caplog.text


def test_regime_allows_trade_logs(caplog):
    ctx = BotContext(positions={}, df_cache={}, regime_cache={}, config={})
    ctx.trading_disabled = True
    ctx.regime = "risk_off"
    caplog.set_level(logging.INFO)
    allowed = main.regime_allows_trade(ctx, "BTC/USD")
    assert not allowed
    assert "suppressed by regime: risk_off" in caplog.text
