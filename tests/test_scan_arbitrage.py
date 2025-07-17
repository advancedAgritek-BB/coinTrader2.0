import asyncio
import types

import crypto_bot.main as main
import crypto_bot.solana as solana_mod
from crypto_bot.utils import market_loader


def dummy_fetch_ticker(price_map, symbol):
    return {"last": price_map[symbol]}


def test_scan_arbitrage_profitable(monkeypatch):
    cex_prices = {"SOL/USDC": 10.0}
    dex_prices = {"SOL/USDC": 11.0}

    async def fake_gecko(*_a, **_k):
        return [], 0.0, dex_prices["SOL/USDC"]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_gecko)
    exchange = types.SimpleNamespace(fetch_ticker=lambda sym: dummy_fetch_ticker(cex_prices, sym))

    cfg = {"arbitrage_pairs": ["SOL/USDC"], "arbitrage_threshold": 0.05}
    res = asyncio.run(main.scan_arbitrage(exchange, cfg))
    assert res == ["SOL/USDC"]


def test_scan_arbitrage_not_profitable(monkeypatch):
    cex_prices = {"SOL/USDC": 10.0}
    dex_prices = {"SOL/USDC": 10.4}

    async def fake_gecko2(*_a, **_k):
        return [], 0.0, dex_prices["SOL/USDC"]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_gecko2)
    exchange = types.SimpleNamespace(fetch_ticker=lambda sym: dummy_fetch_ticker(cex_prices, sym))

    cfg = {"arbitrage_pairs": ["SOL/USDC"], "arbitrage_threshold": 0.05}
    res = asyncio.run(main.scan_arbitrage(exchange, cfg))
    assert res == []
