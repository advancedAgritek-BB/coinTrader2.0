import asyncio
import logging
import pandas as pd
import types, sys
import pytest

from crypto_bot.phase_runner import BotContext
import crypto_bot.main as main

# Provide dummy modules for optional dependencies
_dummy = types.ModuleType("dummy")
for mod in ["telegram", "gspread", "scipy", "scipy.stats", "redis"]:
    sys.modules.setdefault(mod, _dummy)


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
            "benchmark_symbols": [],
        },
    )
    ctx.exchange = object()
    return ctx


def _patch_common(monkeypatch):
    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "calc_atr", lambda df, period=14: pd.Series([0.02]))
    monkeypatch.setattr(main, "compute_average_atr", lambda *_a, **_k: 0.01)
    monkeypatch.setattr(main, "get_market_regime", lambda *_a, **_k: "unknown")

from collections import deque


def test_fetch_candidates_empty_universe_aborts(monkeypatch, caplog):
    ctx = _ctx()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("ETH/USD", 1.0)], []

    _patch_common(monkeypatch)
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)

    caplog.set_level(logging.WARNING)
    with pytest.raises(RuntimeError, match="Active universe is empty"):
        asyncio.run(main.fetch_candidates(ctx))
    assert "configure 'symbols' or adjust filters" in caplog.text


def test_fetch_candidates_empty_universe_warns(monkeypatch, caplog):
    ctx = _ctx()
    ctx.config["abort_on_empty_universe"] = False

    async def fake_get_filtered_symbols(ex, cfg):
        return [("ETH/USD", 1.0)], []

    _patch_common(monkeypatch)
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)

    caplog.set_level(logging.WARNING)
    asyncio.run(main.fetch_candidates(ctx))

    assert ctx.active_universe == []
    assert "Active universe is empty" in caplog.text
