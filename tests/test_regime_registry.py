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
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "key")
    monkeypatch.setenv("CT_MODELS_BUCKET", "models")
    monkeypatch.setenv("CT_REGIME_PREFIX", "models/regime")
    monkeypatch.setenv(
        "CT_REGIME_MODEL_TEMPLATE", "{prefix}/{symbol}/{symbol_lower}_regime_lgbm.pkl"
    )
    monkeypatch.setattr(registry, "_no_model_logged", False)

    caplog.set_level(logging.INFO, logger="crypto_bot.regime.registry")

    blob, meta = registry.load_latest_regime("BTCUSD")

    assert blob == b"direct-bytes"
    assert meta == {}
    assert calls == [
        "models/regime/BTCUSD/LATEST.json",
        "models/regime/BTCUSD/btcusd_regime_lgbm.pkl",
    ]
    assert not [r for r in caplog.records if "No regime model found" in r.message]

