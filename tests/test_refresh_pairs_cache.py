import json
import logging
import sys
import types
import os
import time

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
sys.modules.setdefault("crypto_bot.utils.market_loader", ml_mod)
import tasks.refresh_pairs as rp


def test_refresh_pairs_error_keeps_old_cache(tmp_path, monkeypatch, caplog):
    file = tmp_path / "liquid_pairs.json"
    file.write_text(json.dumps({"ETH/USD": 0}))
    old = time.time() - 7200
    os.utime(file, (old, old))
    os.utime(file, (0, 0))
    monkeypatch.setattr(rp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rp, "PAIR_FILE", file)

    class DummyExchange:
        async def fetch_tickers(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(rp, "get_exchange", lambda cfg: DummyExchange())

    with caplog.at_level(logging.ERROR):
        result = rp.refresh_pairs(10_000_000, 40, {})

    assert result == ["ETH/USD"]
    assert set(json.loads(file.read_text())) == {"ETH/USD"}
    assert any("boom" in r.getMessage() for r in caplog.records)


def test_is_cache_fresh(monkeypatch, tmp_path):
    file = tmp_path / "liquid_pairs.json"
    monkeypatch.setattr(rp, "PAIR_FILE", file)
    assert not rp.is_cache_fresh()
    file.write_text("[]")
    assert rp.is_cache_fresh()
    old = time.time() - 7200
    os.utime(file, (old, old))
    assert not rp.is_cache_fresh()


def test_refresh_pairs_returns_cached_when_fresh(monkeypatch, tmp_path):
    file = tmp_path / "liquid_pairs.json"
    file.write_text(json.dumps(["BTC/USD"]))
    monkeypatch.setattr(rp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rp, "PAIR_FILE", file)

    class BadExchange:
        async def fetch_tickers(self):
            raise RuntimeError("should not be called")

    monkeypatch.setattr(rp, "get_exchange", lambda cfg: BadExchange())
    result = rp.refresh_pairs(0, 40, {}, force_refresh=False)
    assert result == ["BTC/USD"]


def test_refresh_pairs_force_refresh(monkeypatch, tmp_path):
    file = tmp_path / "liquid_pairs.json"
    file.write_text(json.dumps(["OLD/USD"]))
    monkeypatch.setattr(rp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rp, "PAIR_FILE", file)

    class DummyExchange:
        async def fetch_tickers(self):
            return {"NEW/USD": {"quoteVolume": 2_000_000}}

    monkeypatch.setattr(rp, "get_exchange", lambda cfg: DummyExchange())
    async def no_sol(*_a):
        return []
    monkeypatch.setattr(rp, "get_solana_liquid_pairs", no_sol)

    result = rp.refresh_pairs(0, 1, {}, force_refresh=True)
    assert result == ["NEW/USD"]

