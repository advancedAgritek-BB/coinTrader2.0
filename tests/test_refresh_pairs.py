import json
import sys
import types
import asyncio
import pytest
import os
import time

import crypto_bot.utils.http_client  # ensure module present

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
async def _dummy_gecko(*_a, **_k):
    return None
ml_mod.fetch_geckoterminal_ohlcv = _dummy_gecko
ml_mod.get_kraken_listing_date = lambda *_a, **_k: None
ml_mod._is_valid_base_token = lambda *_a, **_k: True
sys.modules.setdefault("crypto_bot.utils.market_loader", ml_mod)

from tasks import refresh_pairs as rp


class DummyExchange:
    def __init__(self, tickers, markets=None):
        self.tickers = tickers
        if markets is None:
            markets = {
                sym: {
                    "active": True,
                    "contract": False,
                    "index": False,
                    "type": "spot",
                    "quote": sym.split("/")[1],
                }
                for sym in tickers
            }
        self.markets = markets
        self.requested = None

    async def load_markets(self):
        return self.markets

    async def fetch_tickers(self, symbols=None):
        self.requested = symbols
        if symbols is None:
            return self.tickers
        return {s: self.tickers[s] for s in symbols if s in self.tickers}


class FailingExchange:
    def __init__(self, markets=None):
        self.markets = markets or {
            "BTC/USD": {
                "active": True,
                "contract": False,
                "index": False,
                "type": "spot",
                "quote": "USD",
            }
        }

    async def load_markets(self):
        return self.markets

    async def fetch_tickers(self, symbols=None):
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


def test_refresh_pairs_filters_market_flags(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    pair_file = cache_dir / "liquid_pairs.json"
    monkeypatch.setattr(rp, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(rp, "PAIR_FILE", pair_file)
    tickers = {
        "GOOD/USD": {"quoteVolume": 5_000_000},
        "BAD1/USD": {"quoteVolume": 5_000_000},
        "BAD2/USD": {"quoteVolume": 5_000_000},
        "BAD3/USD": {"quoteVolume": 5_000_000},
        "EUR/EUR": {"quoteVolume": 5_000_000},
    }
    markets = {
        "GOOD/USD": {
            "active": True,
            "contract": False,
            "index": False,
            "type": "spot",
            "quote": "USD",
        },
        "BAD1/USD": {"active": False, "contract": False, "index": False, "type": "spot", "quote": "USD"},
        "BAD2/USD": {"active": True, "contract": True, "index": False, "type": "spot", "quote": "USD"},
        "BAD3/USD": {"active": True, "contract": False, "index": False, "type": "swap", "quote": "USD"},
        "EUR/EUR": {
            "active": True,
            "contract": False,
            "index": False,
            "type": "spot",
            "quote": "EUR",
        },
    }
    ex = DummyExchange(tickers, markets)
    monkeypatch.setattr(rp, "get_exchange", lambda _cfg: ex)
    async def no_sol(*_a):
        return []
    monkeypatch.setattr(rp, "get_solana_liquid_pairs", no_sol)
    cfg = {"refresh_pairs": {"allowed_quote_currencies": ["USD"]}}
    pairs = rp.refresh_pairs(1_000_000, 10, cfg)
    assert pairs == ["GOOD/USD"]
    assert ex.requested == ["GOOD/USD"]


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
        {"base": {"symbol": "A"}, "quote": {"symbol": "USDC"}, "liquidity": 2_000_000},
        {"base": {"symbol": "B"}, "quote": {"symbol": "USDC"}, "liquidity": 500_000},
        {"base": {"symbol": "C"}, "quote": {"symbol": "OTHER"}, "liquidity": 5_000_000},
    ]
    session = DummySession(data)
    monkeypatch.setattr(rp, "get_session", lambda: session)
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
