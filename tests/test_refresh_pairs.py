import json
import sys
import types
import asyncio
import pytest
import os
import time

sys.modules.setdefault("ccxt", types.ModuleType("ccxt"))
sys.modules.setdefault("ccxt.async_support", types.ModuleType("ccxt.async_support"))
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *_a, **_k: {}
sys.modules.setdefault("yaml", yaml_mod)
ml_mod = types.ModuleType("crypto_bot.utils.market_loader")
ml_mod.timeframe_seconds = lambda *_a, **_k: 60
ml_mod.load_kraken_symbols = lambda *_a, **_k: []
ml_mod.fetch_ohlcv_async = lambda *_a, **_k: None
ml_mod.load_ohlcv = lambda *_a, **_k: None
ml_mod.fetch_order_book_async = lambda *_a, **_k: None
ml_mod.load_ohlcv_parallel = lambda *_a, **_k: None
ml_mod.update_ohlcv_cache = lambda *_a, **_k: None
ml_mod.update_multi_tf_ohlcv_cache = lambda *_a, **_k: None
ml_mod.fetch_geckoterminal_ohlcv = lambda *_a, **_k: None
sys.modules.setdefault("crypto_bot.utils.market_loader", ml_mod)

from tasks import refresh_pairs as rp


class DummyExchange:
    def __init__(self, tickers):
        self.tickers = tickers

    async def fetch_tickers(self):
        return self.tickers


class FailingExchange:
    async def fetch_tickers(self):
        raise Exception("boom")




def test_refresh_pairs_creates_file(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    pair_file = cache_dir / "liquid_pairs.json"
    monkeypatch.setattr(rp, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(rp, "PAIR_FILE", pair_file)
    tickers = {
        "BTC/USD": {"quoteVolume": 3_000_000},
        "ETH/USD": {"quoteVolume": 2_000_000},
        "XRP/USD": {"quoteVolume": 100_000},
    }
    monkeypatch.setattr(rp, "get_exchange", lambda _cfg: DummyExchange(tickers))
    async def no_sol(*_a):
        return []
    monkeypatch.setattr(rp, "get_solana_liquid_pairs", no_sol)
    pairs = rp.refresh_pairs(1_000_000, 2, {})
    assert pairs == ["BTC/USD", "ETH/USD"]
    assert pair_file.exists()
    data = json.loads(pair_file.read_text())
    assert set(data) == {"BTC/USD", "ETH/USD"}


def test_refresh_pairs_fallback(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    pair_file = cache_dir / "liquid_pairs.json"
    cache_dir.mkdir()
    pair_file.write_text(json.dumps({"OLD/USD": 0}))
    monkeypatch.setattr(rp, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(rp, "PAIR_FILE", pair_file)
    monkeypatch.setattr(rp, "get_exchange", lambda _cfg: FailingExchange())
    async def no_sol(*_a):
        return []
    monkeypatch.setattr(rp, "get_solana_liquid_pairs", no_sol)
    pairs = rp.refresh_pairs(1_000_000, 2, {})
    assert pairs == ["OLD/USD"]
    assert set(json.loads(pair_file.read_text())) == {"OLD/USD"}


def test_refresh_pairs_filters_quote(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    pair_file = cache_dir / "liquid_pairs.json"
    monkeypatch.setattr(rp, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(rp, "PAIR_FILE", pair_file)
    tickers = {
        "BTC/EUR": {"quoteVolume": 3_000_000},
        "ETH/USD": {"quoteVolume": 2_000_000},
    }
    monkeypatch.setattr(rp, "get_exchange", lambda _cfg: DummyExchange(tickers))
    async def no_sol(*_a):
        return []
    monkeypatch.setattr(rp, "get_solana_liquid_pairs", no_sol)
    cfg = {"refresh_pairs": {"allowed_quote_currencies": ["USD"]}}
    pairs = rp.refresh_pairs(1_000_000, 2, cfg)
    assert pairs == ["ETH/USD"]
    assert set(json.loads(pair_file.read_text())) == {"ETH/USD"}


def test_refresh_pairs_blacklist(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    pair_file = cache_dir / "liquid_pairs.json"
    monkeypatch.setattr(rp, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(rp, "PAIR_FILE", pair_file)
    tickers = {
        "SCAM/USD": {"quoteVolume": 4_000_000},
        "BTC/USD": {"quoteVolume": 3_000_000},
    }
    monkeypatch.setattr(rp, "get_exchange", lambda _cfg: DummyExchange(tickers))
    async def no_sol(*_a):
        return []
    monkeypatch.setattr(rp, "get_solana_liquid_pairs", no_sol)
    cfg = {"refresh_pairs": {"blacklist_assets": ["SCAM"]}}
    pairs = rp.refresh_pairs(1_000_000, 2, cfg)
    assert pairs == ["BTC/USD"]
    assert set(json.loads(pair_file.read_text())) == {"BTC/USD"}


class DummyResp:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data

    def raise_for_status(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class DummySession:
    def __init__(self, data):
        self.data = data
        self.url = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, url, timeout=10):
        self.url = url
        return DummyResp(self.data)


def test_get_solana_liquid_pairs(monkeypatch):
    data = [
        {"name": "A/USDC", "liquidity": 2_000_000},
        {"name": "B/USDC", "liquidity": 500_000},
        {"name": "C/OTHER", "liquidity": 5_000_000},
    ]
    session = DummySession(data)
    aiohttp_mod = type("M", (), {"ClientSession": lambda: session, "ClientError": Exception})
    monkeypatch.setattr(rp, "aiohttp", aiohttp_mod)
    res = asyncio.run(rp.get_solana_liquid_pairs(1_000_000))
    assert res == ["A/USDC"]
    assert session.url == "https://api.raydium.io/v2/main/pairs"


def test_refresh_pairs_includes_solana(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    pair_file = cache_dir / "liquid_pairs.json"
    monkeypatch.setattr(rp, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(rp, "PAIR_FILE", pair_file)
    tickers = {"BTC/USD": {"quoteVolume": 2_000_000}}
    monkeypatch.setattr(rp, "get_exchange", lambda _cfg: DummyExchange(tickers))

    async def fake_sol(min_vol, quote):
        return [f"SOL/{quote}"]

    monkeypatch.setattr(rp, "get_solana_liquid_pairs", fake_sol)
    pairs = rp.refresh_pairs(1_000_000, 5, {})
    assert "SOL/USDC" in pairs


def test_refresh_pairs_uses_fresh_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    pair_file = cache_dir / "liquid_pairs.json"
    cache_dir.mkdir()
    pair_file.write_text(json.dumps(["BTC/USD", "ETH/USD"]))
    os.utime(pair_file, (time.time(), time.time()))
    monkeypatch.setattr(rp, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(rp, "PAIR_FILE", pair_file)

    class DummyExchange:
        def __init__(self):
            self.called = False

        async def fetch_tickers(self):
            self.called = True
            raise RuntimeError("boom")

    ex = DummyExchange()
    monkeypatch.setattr(rp, "get_exchange", lambda _cfg: ex)
    monkeypatch.setattr(rp, "get_solana_liquid_pairs", lambda *_a: [])

    pairs = rp.refresh_pairs(1_000_000, 40, {})
    assert pairs == ["BTC/USD", "ETH/USD"]
    assert not ex.called


def test_refresh_pairs_respects_onchain_default_quote(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    pair_file = cache_dir / "liquid_pairs.json"
    monkeypatch.setattr(rp, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(rp, "PAIR_FILE", pair_file)
    tickers = {"BTC/USD": {"quoteVolume": 2_000_000}}
    monkeypatch.setattr(rp, "get_exchange", lambda _cfg: DummyExchange(tickers))

    called: dict[str, str] = {}

    async def fake_sol(min_vol, quote):
        called["quote"] = quote
        return [f"SOL/{quote}"]

    monkeypatch.setattr(rp, "get_solana_liquid_pairs", fake_sol)
    cfg = {"onchain_default_quote": "USDT"}
    pairs = rp.refresh_pairs(1_000_000, 5, cfg)
    assert called["quote"] == "USDT"
    assert "SOL/USDT" in pairs
