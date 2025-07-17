import asyncio
from collections import deque
import pandas as pd

from crypto_bot.phase_runner import BotContext
import crypto_bot.main as main


def _setup_ctx():
    df = pd.DataFrame({"high": [1, 2], "low": [0, 1], "close": [1, 2]})
    ctx = BotContext(
        positions={},
        df_cache={"1h": {"BTC/USD": df}},
        regime_cache={},
        config={
            "timeframe": "1h",
            "symbols": ["BTC/USD", "ETH/USD"],
            "symbol_batch_size": 1,
            "symbol_filter": {"min_volume_usd": 1000, "volume_percentile": 10},
        },
    )
    ctx.exchange = object()
    return ctx


def test_fetch_candidates_market_pump(monkeypatch):
    ctx = _setup_ctx()
    ctx.config["symbol_batch_size"] = 2

    async def fake_get_filtered_symbols(ex, cfg):
        if cfg["symbol_filter"]["min_volume_usd"] == 500:
            return [("BTC/USD", 1.0), ("ETH/USD", 0.9)]
        return [("BTC/USD", 1.0)]

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "is_market_pumping", lambda *a, **k: True)
    monkeypatch.setattr(main, "calc_atr", lambda df, window=14: 0.02)

    asyncio.run(main.fetch_candidates(ctx))

    assert len(ctx.current_batch) == 2
    assert ctx.config["symbol_filter"]["min_volume_usd"] == 1000
    assert ctx.config["symbol_filter"]["volume_percentile"] == 10


def test_fetch_candidates_no_pump(monkeypatch):
    ctx = _setup_ctx()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("BTC/USD", 1.0)]

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "is_market_pumping", lambda *a, **k: False)
    monkeypatch.setattr(main, "calc_atr", lambda df, window=14: 0.02)

    asyncio.run(main.fetch_candidates(ctx))

    assert len(ctx.current_batch) == 1
    assert ctx.config["symbol_filter"]["min_volume_usd"] == 1000
    assert ctx.config["symbol_filter"]["volume_percentile"] == 10
