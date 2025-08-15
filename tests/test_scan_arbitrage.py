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

    def fake_gecko(*_a, **_k):
        price = dex_prices["SOL/USDC"]
        return [[0, price, price, price, price, 0]]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_gecko)
    exchange = types.SimpleNamespace(fetch_ticker=lambda sym: dummy_fetch_ticker(cex_prices, sym))

    cfg = {"arbitrage_pairs": ["SOL/USDC"], "arbitrage_threshold": 0.005}
    res = asyncio.run(main.scan_arbitrage(exchange, cfg))
    assert res == ["SOL/USDC"]


def test_scan_arbitrage_not_profitable(monkeypatch):
    cex_prices = {"SOL/USDC": 10.0}
    dex_prices = {"SOL/USDC": 10.04}

    def fake_gecko2(*_a, **_k):
        price = dex_prices["SOL/USDC"]
        return [[0, price, price, price, price, 0]]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_gecko2)
    exchange = types.SimpleNamespace(fetch_ticker=lambda sym: dummy_fetch_ticker(cex_prices, sym))

    cfg = {"arbitrage_pairs": ["SOL/USDC"], "arbitrage_threshold": 0.005}
    res = asyncio.run(main.scan_arbitrage(exchange, cfg))
    assert res == []


def test_scan_cex_arbitrage(monkeypatch):
    ex1_prices = {"SOL/USDC": 10.0}
    ex2_prices = {"SOL/USDC": 11.0}
    ex1 = types.SimpleNamespace(fetch_ticker=lambda sym: dummy_fetch_ticker(ex1_prices, sym))
    ex2 = types.SimpleNamespace(fetch_ticker=lambda sym: dummy_fetch_ticker(ex2_prices, sym))
    cfg = {"arbitrage_pairs": ["SOL/USDC"], "arbitrage_threshold": 0.05}
    res = asyncio.run(main.scan_cex_arbitrage(ex1, ex2, cfg))
    assert res == ["SOL/USDC"]
