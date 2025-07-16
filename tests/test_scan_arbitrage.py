import asyncio
import types

import crypto_bot.main as main
import crypto_bot.solana as solana_mod


def dummy_fetch_ticker(price_map, symbol):
    return {"last": price_map[symbol]}


def test_scan_arbitrage_profitable(monkeypatch):
    cex_prices = {"SOL/USDC": 10.0}
    dex_prices = {"SOL/USDC": 11.0}

    async def fake_prices(syms):
        return dex_prices

    monkeypatch.setattr(solana_mod, "fetch_solana_prices", fake_prices)
    exchange = types.SimpleNamespace(fetch_ticker=lambda sym: dummy_fetch_ticker(cex_prices, sym))

    cfg = {"arbitrage_pairs": ["SOL/USDC"], "arbitrage_threshold": 0.05}
    res = asyncio.run(main.scan_arbitrage(exchange, cfg))
    assert res == ["SOL/USDC"]


def test_scan_arbitrage_not_profitable(monkeypatch):
    cex_prices = {"SOL/USDC": 10.0}
    dex_prices = {"SOL/USDC": 10.4}

    async def fake_prices2(syms):
        return dex_prices

    monkeypatch.setattr(solana_mod, "fetch_solana_prices", fake_prices2)
    exchange = types.SimpleNamespace(fetch_ticker=lambda sym: dummy_fetch_ticker(cex_prices, sym))

    cfg = {"arbitrage_pairs": ["SOL/USDC"], "arbitrage_threshold": 0.05}
    res = asyncio.run(main.scan_arbitrage(exchange, cfg))
    assert res == []
