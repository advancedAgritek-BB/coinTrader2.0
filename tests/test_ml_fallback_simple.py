import pandas as pd
from crypto_bot.regime import regime_classifier as rc


def _make_df(rows: int = 30) -> pd.DataFrame:
    close = pd.Series(range(rows)).astype(float)
    return pd.DataFrame({
        "open": close,
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "volume": [100] * rows,
    })


def test_no_ml_fallback_when_disabled(monkeypatch, tmp_path):
    df = _make_df()
    monkeypatch.setattr(rc, "_classify_core", lambda *_a, **_k: "unknown")
    called = False

    def fake_ml(df):
        nonlocal called
        called = True
        return "trending", 0.9

    monkeypatch.setattr(rc, "_ml_fallback", fake_ml)

    cfg = tmp_path / "regime.yaml"
    cfg.write_text("use_ml_regime_classifier: false\nml_min_bars: 5\n")

    label, _ = rc.classify_regime(df, config_path=str(cfg))
    assert label == "unknown"
    assert not called


def test_ml_fallback_triggers(monkeypatch, tmp_path):
    df = _make_df()
    monkeypatch.setattr(rc, "_classify_core", lambda *_a, **_k: "unknown")
    called = False

    def fake_ml(df):
        nonlocal called
        called = True
        return "trending", 0.8

    monkeypatch.setattr(rc, "_ml_fallback", fake_ml)

    cfg = tmp_path / "regime.yaml"
    cfg.write_text(
        "use_ml_regime_classifier: true\nml_min_bars: 5\nml_blend_weight: 1.0\n"
    )

    label, probs = rc.classify_regime(df, config_path=str(cfg))
    assert called
    assert label == "trending"
    assert probs["trending"] == 0.8


def test_corrupted_model_data(monkeypatch):
    from crypto_bot.regime import ml_fallback as mf

    df = _make_df()
    mf._model = None

    monkeypatch.setattr(
        mf.base64, "b64decode", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad"))
    )

    model = mf.load_model()
    assert model is not None
    label, conf = mf.predict_regime(df)
    assert label != "unknown"
    assert conf > 0.0


def test_joblib_failure_returns_stub_model(monkeypatch):
    from crypto_bot.regime import ml_fallback as mf

    df = _make_df()
    mf._model = None

    def bad_load(*_a, **_k):
        raise RuntimeError("bad model")

    monkeypatch.setattr(mf.joblib, "load", bad_load)

    model = mf.load_model()
    assert model is not None
    label, conf = mf.predict_regime(df)
    assert label != "unknown"
    assert conf > 0.0
