import pickle
from pathlib import Path

from crypto_bot.ml import model_loader as ml


def test_load_regime_model_explicit_path(monkeypatch, tmp_path, caplog):
    model_obj = {"model": "dummy"}
    model_file = tmp_path / "xrpusd_regime_lgbm.pkl"
    model_file.write_bytes(pickle.dumps(model_obj))

    monkeypatch.setenv("CT_MODEL_LOCAL_PATH", str(model_file))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("CT_MODEL_FALLBACK_URL", raising=False)

    model, scaler, path = ml.load_regime_model("XRPUSD")
    assert model == "dummy"
    assert path == str(model_file)


def test_load_regime_model_explicit_dir(monkeypatch, tmp_path, caplog):
    model_obj = {"model": "dummy"}
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "xrpusd_regime_lgbm.pkl"
    model_file.write_bytes(pickle.dumps(model_obj))

    monkeypatch.setenv("CT_MODEL_LOCAL_PATH", str(model_dir))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("CT_MODEL_FALLBACK_URL", raising=False)

    model, scaler, path = ml.load_regime_model("XRPUSD")
    assert model == "dummy"
    assert path == str(model_file)


def test_load_regime_model_search_dirs(monkeypatch, caplog):
    monkeypatch.delenv("CT_MODEL_LOCAL_PATH", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("CT_MODEL_FALLBACK_URL", raising=False)

    model_obj = {"model": "dummy"}
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "crypto_bot" / "models"
    model_dir.mkdir(exist_ok=True)
    model_file = model_dir / "xrpusd_regime_lgbm.pkl"
    model_file.write_bytes(pickle.dumps(model_obj))

    try:
        model, scaler, path = ml.load_regime_model("XRPUSD")
        assert model == "dummy"
        assert path == str(model_file)
    finally:
        model_file.unlink(missing_ok=True)
