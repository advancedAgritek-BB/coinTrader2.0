import json
import sys
import types
import asyncio
import pytest

sys.modules.setdefault("ccxt", types.ModuleType("ccxt"))
sys.modules.setdefault("ccxt.async_support", types.ModuleType("ccxt.async_support"))
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *_a, **_k: {}
sys.modules.setdefault("yaml", yaml_mod)

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
    pairs = rp.refresh_pairs(1_000_000, 2, {})
    assert pairs == ["BTC/USD", "ETH/USD"]
    assert pair_file.exists()
    assert json.loads(pair_file.read_text()) == ["BTC/USD", "ETH/USD"]


def test_refresh_pairs_fallback(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    pair_file = cache_dir / "liquid_pairs.json"
    cache_dir.mkdir()
    pair_file.write_text(json.dumps(["OLD/USD"]))
    monkeypatch.setattr(rp, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(rp, "PAIR_FILE", pair_file)
    monkeypatch.setattr(rp, "get_exchange", lambda _cfg: FailingExchange())
    pairs = rp.refresh_pairs(1_000_000, 2, {})
    assert pairs == ["OLD/USD"]
    assert json.loads(pair_file.read_text()) == ["OLD/USD"]
