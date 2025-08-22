import pickle
import logging
from pathlib import Path

from crypto_bot import main
from crypto_bot.regime import regime_classifier as rc


def test_load_config_detects_local_model(monkeypatch, tmp_path, caplog):
    model_path = Path("crypto_bot/models/xrpusd_regime_lgbm.pkl")
    model_path.write_bytes(pickle.dumps({"model": "dummy"}))
    try:
        rc._supabase_model = None
        rc._supabase_scaler = None
        rc._supabase_symbol = None
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("ml_enabled: true\nsymbol: XRPUSD\n")
        monkeypatch.setattr(main, "CONFIG_PATH", cfg_path)
        main._CONFIG_CACHE.clear()
        main._CONFIG_MTIMES.clear()
        main._LAST_ML_CFG = None
        monkeypatch.setattr(main.ml_utils, "init_ml_components", lambda: (True, ""))
        monkeypatch.setattr(main.ml_utils, "ML_AVAILABLE", True)
        caplog.set_level(logging.INFO, logger="bot")
        main.load_config()
        assert "Loaded global regime model for XRPUSD from local file" in caplog.text
    finally:
        model_path.unlink(missing_ok=True)
