import json
import logging
import sys
import types

sys.modules.setdefault("ccxt", types.ModuleType("ccxt"))
sys.modules.setdefault("ccxt.async_support", types.ModuleType("ccxt.async_support"))
import tasks.refresh_pairs as rp


def test_refresh_pairs_error_keeps_old_cache(tmp_path, monkeypatch, caplog):
    file = tmp_path / "liquid_pairs.json"
    file.write_text(json.dumps({"ETH/USD": 0}))
    monkeypatch.setattr(rp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rp, "PAIR_FILE", file)

    class DummyExchange:
        async def fetch_tickers(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(rp, "get_exchange", lambda cfg: DummyExchange())

    with caplog.at_level(logging.ERROR):
        result = rp.refresh_pairs(1_000_000, 40, {})

    assert result == ["ETH/USD"]
    assert set(json.loads(file.read_text())) == {"ETH/USD"}
    assert any("boom" in r.getMessage() for r in caplog.records)

