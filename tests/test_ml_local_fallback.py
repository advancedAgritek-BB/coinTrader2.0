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

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setattr(main, "ML_AVAILABLE", True)
    monkeypatch.setenv("CT_MODEL_FALLBACK_URL", remote.resolve().as_uri())

    cfg = {
        "ml_enabled": True,
        "symbol": "XRPUSD",
    }

    caplog.set_level(logging.INFO, logger="bot")
    main._ensure_ml_if_needed(cfg)
    assert f"Loaded global regime model for XRPUSD from {remote.resolve().as_uri()}" in caplog.text
