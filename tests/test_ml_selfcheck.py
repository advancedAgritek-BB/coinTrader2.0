import importlib  # needed for reload_selfcheck fixture
import logging

import pytest

from crypto_bot.ml import selfcheck


@pytest.fixture(autouse=True)
def reload_selfcheck():
    importlib.reload(selfcheck)


def _clear_supabase_env(monkeypatch):
    for var in [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_KEY",
        "SUPABASE_API_KEY",
        "SUPABASE_ANON_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)


def test_log_ml_status_once_logs_supabase_status(monkeypatch, caplog):
    _clear_supabase_env(monkeypatch)
    monkeypatch.setattr(selfcheck, "_REQUIRED_PACKAGES", ())
    caplog.set_level(logging.INFO, logger="crypto_bot.ml")

    selfcheck.log_ml_status_once()
    assert (
        "ML status: packages=True supabase_url=False key_present=False"
        in caplog.text
    )

    caplog.clear()
    selfcheck.log_ml_status_once()
    assert "ML status" not in caplog.text


def test_log_ml_status_once_detects_credentials(monkeypatch, caplog):
    _clear_supabase_env(monkeypatch)
    monkeypatch.setattr(selfcheck, "_REQUIRED_PACKAGES", ())
    monkeypatch.setenv("SUPABASE_URL", "http://example")
    monkeypatch.setenv("SUPABASE_API_KEY", "x")
    caplog.set_level(logging.INFO, logger="crypto_bot.ml")

    selfcheck.log_ml_status_once()
    assert (
        "ML status: packages=True supabase_url=True key_present=True"
        in caplog.text
    )
