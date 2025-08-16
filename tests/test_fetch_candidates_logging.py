import asyncio
from collections import deque
import logging
import pandas as pd
import types, sys

from crypto_bot.phase_runner import BotContext
dummy = types.ModuleType("dummy")
for mod in ["telegram", "gspread", "scipy", "scipy.stats", "redis"]:
    sys.modules.setdefault(mod, dummy)
for mod in ["solana", "solana.rpc", "solana.rpc.async_api"]:
    sys.modules.setdefault(mod, dummy)
dummy.AsyncClient = object
sys.modules.setdefault("keyring", dummy)

import crypto_bot.main as main


def test_fetch_candidates_logs_batch(monkeypatch, caplog):
    df = pd.DataFrame({"high": [1, 2], "low": [0, 1], "close": [1, 2]})
    ctx = BotContext(
        positions={},
        df_cache={"1h": {"BTC/USD": df}},
        regime_cache={},
        config={"timeframe": "1h", "symbols": ["BTC/USD"], "symbol_batch_size": 1},
    )
    ctx.exchange = object()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("BTC/USD", 1.0)], []

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "calc_atr", lambda df, window=14: 0.02)
    monkeypatch.setattr(main, "get_market_regime", lambda *_a, **_k: "trending")
    monkeypatch.setattr(main, "compute_average_atr", lambda *_a, **_k: 0.01)

    main.logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG)

    asyncio.run(main.fetch_candidates(ctx))

    assert ctx.current_batch == ["BTC/USDT"]
    assert "Current batch" in caplog.text


def test_fetch_candidates_dedupes_and_filters(monkeypatch):
    df = pd.DataFrame({"high": [1, 2], "low": [0, 1], "close": [1, 2]})
    ctx = BotContext(
        positions={},
        df_cache={"1h": {"BTC/USDT": df}},
        regime_cache={},
        config={"timeframe": "1h", "symbols": ["BTC/USDT"], "symbol_batch_size": 5},
    )

    class DummyExchange:
        def list_markets(self):
            return ["BTC/USDT", "ETH/USDT"]

    ctx.exchange = DummyExchange()

    async def fake_get_filtered_symbols(ex, cfg):
        return [
            ("ETH/USDT", 2.0),
            ("BTC/USDT", 1.0),
            ("DOGE/USDT", 0.5),
            ("ETH/USDT", 0.1),
        ], []

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "calc_atr", lambda df, window=14: 0.02)
    monkeypatch.setattr(main, "get_market_regime", lambda *_a, **_k: "unknown")
    monkeypatch.setattr(main, "compute_average_atr", lambda *_a, **_k: 0.01)

    asyncio.run(main.fetch_candidates(ctx))

    assert ctx.current_batch == ["BTC/USDT", "ETH/USDT"]
    assert len(ctx.current_batch) == len(set(ctx.current_batch))
