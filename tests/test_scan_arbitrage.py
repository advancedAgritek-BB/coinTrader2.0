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
        price = dex_prices["SOL/USDC"]
        return [[0, price, price, price, price, 0]]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_gecko)
    monkeypatch.setattr(main, "fetch_geckoterminal_ohlcv", fake_gecko)
    exchange = types.SimpleNamespace(fetch_ticker=lambda sym: dummy_fetch_ticker(cex_prices, sym))

    cfg = {"arbitrage_pairs": ["SOL/USDC"], "arbitrage_threshold": 0.005}
    res = asyncio.run(main.scan_arbitrage(exchange, cfg))
    assert res == ["SOL/USDC"]


def test_scan_arbitrage_not_profitable(monkeypatch):
    cex_prices = {"SOL/USDC": 10.0}
    dex_prices = {"SOL/USDC": 10.04}

    async def fake_gecko2(*_a, **_k):
        price = dex_prices["SOL/USDC"]
        return [[0, price, price, price, price, 0]]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_gecko2)
    monkeypatch.setattr(main, "fetch_geckoterminal_ohlcv", fake_gecko2)
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


def test_scan_cex_arbitrage_bulk(monkeypatch):
    ex1_prices = {"SOL/USDC": 10.0}
    ex2_prices = {"SOL/USDC": 11.0}

    class Ex:
        def __init__(self, prices):
            self.prices = prices
            self.has = {"fetchTickers": True}

        async def fetch_tickers(self, pairs):
            return {sym: {"last": self.prices[sym]} for sym in pairs}

        def fetch_ticker(self, _sym):  # pragma: no cover - should not be called
            raise AssertionError("fetch_ticker should not be used when fetch_tickers is available")

    ex1 = Ex(ex1_prices)
    ex2 = Ex(ex2_prices)
    cfg = {"arbitrage_pairs": ["SOL/USDC"], "arbitrage_threshold": 0.05}
    res = asyncio.run(main.scan_cex_arbitrage(ex1, ex2, cfg))
    assert res == ["SOL/USDC"]
def test_scan_arbitrage_fetch_tickers(monkeypatch):
    cex_prices = {"SOL/USDC": 10.0, "BTC/USDC": 11.5}
    dex_prices = {"SOL/USDC": 11.0, "BTC/USDC": 12.0}

    async def fake_gecko(sym, *_a, **_k):
        price = dex_prices[sym]
        return [[0, price, price, price, price, 0]]

    monkeypatch.setattr(market_loader, "fetch_geckoterminal_ohlcv", fake_gecko)
    monkeypatch.setattr(main, "fetch_geckoterminal_ohlcv", fake_gecko)

    fetch_tickers_called: list[list[str]] = []
    fetch_ticker_called: list[str] = []

    def fetch_tickers(symbols):
        fetch_tickers_called.append(symbols)
        return {"SOL/USDC": {"last": cex_prices["SOL/USDC"]}}

    def fetch_ticker(symbol):
        fetch_ticker_called.append(symbol)
        return {"last": cex_prices[symbol]}

    exchange = types.SimpleNamespace(
        has={"fetchTickers": True},
        fetch_tickers=fetch_tickers,
        fetch_ticker=fetch_ticker,
    )

    cfg = {
        "arbitrage_pairs": ["SOL/USDC", "BTC/USDC"],
        "arbitrage_threshold": 0.05,
    }
    res = asyncio.run(main.scan_arbitrage(exchange, cfg))
    assert res == ["SOL/USDC"]
    assert fetch_tickers_called == [cfg["arbitrage_pairs"]]
    assert fetch_ticker_called == ["BTC/USDC"]
