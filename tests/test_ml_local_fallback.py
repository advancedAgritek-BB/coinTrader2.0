import logging
import pickle
import urllib.request

from crypto_bot import main
from crypto_bot.regime import regime_classifier as rc


def test_ensure_ml_uses_cached_fallback(monkeypatch, tmp_path, caplog):
    model_obj = {"model": "dummy"}
    remote = tmp_path / "remote.pkl"
    remote.write_bytes(pickle.dumps(model_obj))
    cache = tmp_path / "cache.pkl"
    rc._supabase_model = None
    rc._supabase_scaler = None
    rc._supabase_symbol = None

    async def fail_load(_symbol):
        raise Exception("supabase down")

    monkeypatch.setattr(rc, "load_regime_model", fail_load)
    monkeypatch.setattr(main, "ML_AVAILABLE", True)

    cfg = {
        "ml_enabled": True,
        "symbol": "XRPUSD",
        "model_fallback_url": remote.resolve().as_uri(),
        "model_local_path": str(cache),
    }

    caplog.set_level(logging.INFO, logger="bot")
    main._ensure_ml(cfg)
    assert cache.exists()
    assert "Loaded fallback regime model for XRPUSD" in caplog.text

    def urlopen_fail(_url):
        raise Exception("offline")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_fail)
    caplog.clear()
    main._ensure_ml(cfg)
    assert "Loaded cached fallback regime model for XRPUSD" in caplog.text
