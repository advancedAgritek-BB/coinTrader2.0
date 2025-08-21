import logging
import sys
import types

from crypto_bot.regime import registry


def test_loads_direct_model_when_latest_missing(monkeypatch, caplog):
    class Response:
        status_code = 404

    class NotFound(Exception):
        def __init__(self):
            self.response = Response()

    calls = []

    class Bucket:
        def download(self, key):
            calls.append(key)
            if key.endswith("LATEST.json"):
                raise NotFound()
            return b"direct-bytes"

    class Storage:
        def from_(self, _bucket):
            return Bucket()

    class Client:
        storage = Storage()

    def create_client(url, key):  # noqa: D401 - stub
        return Client()

    monkeypatch.setitem(
        sys.modules, "supabase", types.SimpleNamespace(create_client=create_client)
    )
    monkeypatch.setenv("SUPABASE_URL", "http://example")
    monkeypatch.setenv("SUPABASE_KEY", "key")
    monkeypatch.setenv("CT_MODELS_BUCKET", "models")
    monkeypatch.setenv("CT_REGIME_PREFIX", "regime")
    monkeypatch.setenv(
        "CT_REGIME_MODEL_TEMPLATE", "{prefix}/{symbol}/{symbol_lower}_regime_lgbm.pkl"
    )
    monkeypatch.setattr(registry, "_no_model_logged", False)

    caplog.set_level(logging.INFO, logger="crypto_bot.regime.registry")

    blob, meta = registry.load_latest_regime("BTCUSD")

    assert blob == b"direct-bytes"
    assert meta == {}
    assert calls == [
        "regime/BTCUSD/LATEST.json",
        "regime/BTCUSD/btcusd_regime_lgbm.pkl",
    ]
    assert not [r for r in caplog.records if "No regime model found" in r.message]


def test_http_fallback_used_when_supabase_unavailable(monkeypatch, tmp_path, caplog):
    data = b"fallback-bytes"
    remote = tmp_path / "model.pkl"
    remote.write_bytes(data)

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setenv("CT_MODEL_FALLBACK_URL", remote.as_uri())
    monkeypatch.setattr(registry, "_no_model_logged", False)

    def fail_fallback():  # pragma: no cover - ensure not called
        raise AssertionError("heuristic fallback should not be used")

    monkeypatch.setattr(registry, "_load_fallback", fail_fallback)

    caplog.set_level(logging.INFO, logger="crypto_bot.regime.registry")

    blob, meta = registry.load_latest_regime("BTCUSD")

    assert blob == data
    assert meta.get("source") == remote.as_uri()
    assert not [r for r in caplog.records if "No regime model found" in r.message]


def test_http_fallback_logs_and_uses_heuristic_when_download_fails(
    monkeypatch, caplog
):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("CT_MODEL_FALLBACK_URL", "file:///nonexistent.pkl")

    sentinel = object()

    def fake_fallback():
        return sentinel

    monkeypatch.setattr(registry, "_load_fallback", fake_fallback)
    monkeypatch.setattr(registry, "_no_model_logged", False)

    caplog.set_level(logging.INFO, logger="crypto_bot.regime.registry")

    blob, meta = registry.load_latest_regime("BTCUSD")

    assert blob is sentinel
    assert meta == {}
    assert [r for r in caplog.records if "Failed to download fallback model" in r.message]
    assert [r for r in caplog.records if "No regime model found" in r.message]


def test_config_url_unreachable_uses_heuristic(monkeypatch, caplog):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("CT_MODEL_FALLBACK_URL", raising=False)

    import urllib.request
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url: (_ for _ in ()).throw(OSError("network down")),
    )

    monkeypatch.setitem(
        sys.modules,
        "crypto_bot.main",
        types.SimpleNamespace(_LAST_ML_CFG={"model_fallback_url": "http://example"}),
    )

    sentinel = object()

    monkeypatch.setattr(registry, "_load_fallback", lambda: sentinel)
    monkeypatch.setattr(registry, "_no_model_logged", False)

    caplog.set_level(logging.INFO, logger="crypto_bot.regime.registry")

    blob, meta = registry.load_latest_regime("BTCUSD")

    assert blob is sentinel
    assert meta == {}
    assert [r for r in caplog.records if "Failed to download fallback model" in r.message]
    assert [r for r in caplog.records if "No regime model found" in r.message]

