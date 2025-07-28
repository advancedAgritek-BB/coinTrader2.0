import pytest
import sys
import types

class DummyEx:
    def __init__(self, params):
        self.params = params


def test_load_exchange_uses_keyring(monkeypatch):
    sys.modules.setdefault("gspread", types.ModuleType("gspread"))
    sys.modules.setdefault(
        "oauth2client.service_account",
        types.SimpleNamespace(ServiceAccountCredentials=object),
    )
    from crypto_bot.execution import executor

    monkeypatch.setattr(
        executor.ccxt,
        "binance",
        lambda params: DummyEx(params),
        raising=False,
    )
    monkeypatch.setattr(
        executor,
        "keyring",
        type(
            "K",
            (),
            {
                "get_password": lambda service, name: {
                    "api_key": "k",
                    "api_secret": "s",
                }.get(name)
            },
        ),
    )
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_SECRET", raising=False)

    ex = executor.load_exchange()
    assert ex.params["apiKey"] == "k"
    assert ex.params["secret"] == "s"


def test_load_exchange_missing(monkeypatch):
    sys.modules.setdefault("gspread", types.ModuleType("gspread"))
    sys.modules.setdefault(
        "oauth2client.service_account",
        types.SimpleNamespace(ServiceAccountCredentials=object),
    )
    from crypto_bot.execution import executor

    monkeypatch.setattr(
        executor,
        "keyring",
        type("K", (), {"get_password": lambda *a, **k: None}),
    )
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_SECRET", raising=False)

    with pytest.raises(ValueError):
        executor.load_exchange()
