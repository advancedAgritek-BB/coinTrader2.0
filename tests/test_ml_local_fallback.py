import logging
import pickle

from crypto_bot import main
from crypto_bot.regime import regime_classifier as rc


def test_ensure_ml_uses_fallback_url(monkeypatch, tmp_path, caplog):
    model_obj = {"model": "dummy"}
    remote = tmp_path / "remote.pkl"
    remote.write_bytes(pickle.dumps(model_obj))
    rc._supabase_model = None
    rc._supabase_scaler = None
    rc._supabase_symbol = None

    def fail_load(_symbol):
        raise Exception("supabase down")

    monkeypatch.setattr(main, "load_regime_model", fail_load)
    monkeypatch.setattr(main, "ML_AVAILABLE", True)

    cfg = {
        "ml_enabled": True,
        "symbol": "XRPUSD",
        "model_fallback_url": remote.resolve().as_uri(),
    }

    caplog.set_level(logging.INFO, logger="bot")
    main._ensure_ml_if_needed(cfg)
    assert "Loaded fallback regime model for XRPUSD" in caplog.text


def test_model_none_triggers_fallback(monkeypatch, tmp_path, caplog):
    model_obj = {"model": "dummy"}
    remote = tmp_path / "remote.pkl"
    remote.write_bytes(pickle.dumps(model_obj))
    rc._supabase_model = None
    rc._supabase_scaler = None
    rc._supabase_symbol = None

    def return_none(_symbol):
        return None, None, None

    monkeypatch.setattr(main, "load_regime_model", return_none)
    monkeypatch.setattr(main, "ML_AVAILABLE", True)
    main._LAST_ML_CFG = None

    cfg = {
        "ml_enabled": True,
        "symbol": "XRPUSD",
        "model_fallback_url": remote.resolve().as_uri(),
    }

    caplog.set_level(logging.INFO, logger="bot")
    main._ensure_ml_if_needed(cfg)
    assert "Loaded fallback regime model for XRPUSD" in caplog.text
