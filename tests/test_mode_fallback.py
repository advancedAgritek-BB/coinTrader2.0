import asyncio
from collections import deque

import pytest

from crypto_bot import main
from crypto_bot.utils import symbol_utils


@pytest.mark.asyncio
async def test_get_filtered_symbols_cex_fallback(monkeypatch):
    async def fake_filter(exchange, symbols, config):
        return [(symbols[0], 0.0)], [
            (s, 0.0) for s in config.get("onchain_symbols", [])
        ]

    monkeypatch.setattr(symbol_utils, "filter_symbols", fake_filter)

    cfg = {
        "mode": "auto",
        "symbols": ["BTC/USD"],
        "onchain_symbols": ["SOL/USDC"],
        "solana_scanner": {"enabled": False},
    }

    scored, onchain = await symbol_utils.get_filtered_symbols(object(), cfg)

    assert cfg["mode"] == "cex"
    assert [s for s, _ in scored] == ["BTC/USD"]
    assert onchain == ["SOL/USDC"]


@pytest.mark.asyncio
async def test_fetch_candidates_skips_onchain_when_cex(monkeypatch):
    class DummyExchange:
        def list_markets(self):
            return {"BTC/USD": {}, "FOO/USDC": {}, "SOL/USDC": {}}

    ctx = main.BotContext()
    ctx.exchange = DummyExchange()
    ctx.config = {
        "mode": "auto",
        "symbols": ["BTC/USD"],
        "onchain_symbols": ["FOO/USDC"],
        "solana_scanner": {"enabled": False},
    }
    ctx.df_cache = {}
    ctx.risk_manager = None
    ctx.timing = {}

    main.symbol_priority_queue = deque()

    monkeypatch.setattr(main, "is_market_pumping", lambda *a, **k: False)

    async def fake_get_filtered_symbols(exchange, config):
        config["mode"] = "cex"
        return [("BTC/USD", 0.0)], ["FOO/USDC"]

    async def async_empty(*a, **k):
        return {}

    async def async_list(*a, **k):
        return []

    monkeypatch.setattr(main, "get_filtered_symbols", fake_get_filtered_symbols)
    monkeypatch.setattr(main, "fetch_from_helius", async_empty)
    monkeypatch.setattr(main, "compute_average_atr", lambda *a, **k: 0.01)
    monkeypatch.setattr(main, "get_market_regime", lambda *a, **k: asyncio.sleep(0, result="unknown"))
    monkeypatch.setattr(main, "scan_cex_arbitrage", async_list)
    monkeypatch.setattr(main, "scan_arbitrage", async_list)
    monkeypatch.setattr(main, "get_solana_new_tokens", async_list)

    await main.fetch_candidates(ctx)

    assert "FOO/USDC" not in ctx.active_universe
    assert "FOO/USDC" not in ctx.current_batch
