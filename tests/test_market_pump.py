import asyncio
from collections import deque
import pandas as pd

from crypto_bot.phase_runner import BotContext
import types, sys
dummy = types.ModuleType("dummy")
for mod in ["telegram", "gspread", "scipy", "scipy.stats", "redis"]:
    sys.modules.setdefault(mod, dummy)
oauth_module = types.ModuleType("oauth2client")
oauth_module.service_account = types.SimpleNamespace(ServiceAccountCredentials=object)
sys.modules.setdefault("oauth2client", oauth_module)
sys.modules.setdefault("oauth2client.service_account", oauth_module.service_account)
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
            return [("BTC/USD", 1.0), ("ETH/USD", 0.9)], []
        return [("BTC/USD", 1.0)], []

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "is_market_pumping", lambda *a, **k: True)
    monkeypatch.setattr(main, "calc_atr", lambda df, period=14: 0.02)

    asyncio.run(main.fetch_candidates(ctx))

    assert len(ctx.current_batch) == 2
    assert ctx.config["symbol_filter"]["min_volume_usd"] == 1000
    assert ctx.config["symbol_filter"]["volume_percentile"] == 10


def test_fetch_candidates_no_pump(monkeypatch):
    ctx = _setup_ctx()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("BTC/USD", 1.0)], []

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "is_market_pumping", lambda *a, **k: False)
    monkeypatch.setattr(main, "calc_atr", lambda df, period=14: 0.02)

    asyncio.run(main.fetch_candidates(ctx))

    assert len(ctx.current_batch) == 1
    assert ctx.config["symbol_filter"]["min_volume_usd"] == 1000
    assert ctx.config["symbol_filter"]["volume_percentile"] == 10


def test_pump_allocates_solana(monkeypatch):
    ctx = _setup_ctx()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("BTC/USD", 1.0)], []

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "is_market_pumping", lambda *a, **k: True)
    monkeypatch.setattr(main, "calc_atr", lambda df, period=14: 0.02)
    async def fake_regime(*_a, **_k):
        return "trending"

    monkeypatch.setattr(main, "get_market_regime", fake_regime)
    async def fake_scan(*a, **k):
        return []

    monkeypatch.setattr(main, "scan_arbitrage", fake_scan)

    class DummyRM:
        def __init__(self):
            self.weights = None

        def update_allocation(self, w):
            self.weights = w

    ctx.risk_manager = DummyRM()

    asyncio.run(main.fetch_candidates(ctx))

    assert ctx.config["mode"] == "auto"
    assert ctx.risk_manager.weights.get("sniper_solana") == 0.6


def test_trending_queues_arb(monkeypatch):
    ctx = _setup_ctx()

    async def fake_get_filtered_symbols(ex, cfg):
        return [("BTC/USD", 1.0)], []

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "is_market_pumping", lambda *a, **k: False)
    monkeypatch.setattr(main, "calc_atr", lambda df, period=14: 0.02)
    async def fake_regime(*_a, **_k):
        return "trending"

    monkeypatch.setattr(main, "get_market_regime", fake_regime)

    async def fake_scan(*a, **k):
        return ["ARB/USD"]

    monkeypatch.setattr(main, "scan_arbitrage", fake_scan)

    asyncio.run(main.fetch_candidates(ctx))

    assert "ARB/USD" not in ctx.current_batch


def test_volatile_queues_solana(monkeypatch):
    ctx = _setup_ctx()
    ctx.config["solana_scanner"] = {"enabled": True}

    async def fake_get_filtered_symbols(ex, cfg):
        return [("BTC/USD", 1.0)], []

    monkeypatch.setattr(main, "symbol_priority_queue", deque())
    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "is_market_pumping", lambda *a, **k: False)
    monkeypatch.setattr(main, "calc_atr", lambda df, period=14: 0.02)
    async def fake_regime2(*_a, **_k):
        return "volatile"

    monkeypatch.setattr(main, "get_market_regime", fake_regime2)

    async def fake_scan(*a, **k):
        return []

    async def fake_tokens(cfg):
        return ["SOL/USDC"]

    monkeypatch.setattr(main, "scan_arbitrage", fake_scan)
    monkeypatch.setattr(main, "get_solana_new_tokens", fake_tokens)

    asyncio.run(main.fetch_candidates(ctx))

    assert "SOL/USDC" not in ctx.current_batch
