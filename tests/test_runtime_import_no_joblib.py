def test_runtime_import_without_joblib(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "joblib", None)
    from crypto_bot.regime import api  # noqa
    assert hasattr(api, "predict")
