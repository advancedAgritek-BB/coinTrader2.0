import json
import logging
import sys
import types
import os

sys.modules.setdefault("ccxt", types.ModuleType("ccxt"))
sys.modules.setdefault("ccxt.async_support", types.ModuleType("ccxt.async_support"))
ml_mod = types.ModuleType("crypto_bot.utils.market_loader")
ml_mod.timeframe_seconds = lambda *_a, **_k: 60
ml_mod.load_kraken_symbols = lambda *_a, **_k: []
ml_mod.fetch_ohlcv_async = lambda *_a, **_k: None
ml_mod.fetch_order_book_async = lambda *_a, **_k: None
ml_mod.load_ohlcv_parallel = lambda *_a, **_k: None
ml_mod.update_ohlcv_cache = lambda *_a, **_k: None
ml_mod.fetch_geckoterminal_ohlcv = lambda *_a, **_k: None
sys.modules.setdefault("crypto_bot.utils.market_loader", ml_mod)
import tasks.refresh_pairs as rp


def test_refresh_pairs_error_keeps_old_cache(tmp_path, monkeypatch, caplog):
    file = tmp_path / "liquid_pairs.json"
    file.write_text(json.dumps({"ETH/USD": 0}))
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

