"""Tests ML self-check logging for Supabase credentials."""

import importlib
import logging

import pytest

from crypto_bot.ml import selfcheck


@pytest.fixture(autouse=True)
def reload_selfcheck():
    importlib.reload(selfcheck)


def test_log_ml_status_once_logs_supabase_status(monkeypatch, caplog):
    for var in [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_KEY",
        "SUPABASE_API_KEY",
        "SUPABASE_ANON_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)
    caplog.set_level(logging.INFO, logger="crypto_bot.ml")

    selfcheck.log_ml_status_once()
    assert "supabase_url=False" in caplog.text
    assert "key_present=False" in caplog.text

    caplog.clear()
    selfcheck.log_ml_status_once()
    assert "ML status" not in caplog.text
