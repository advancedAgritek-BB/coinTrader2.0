import importlib
import logging

import pytest

import crypto_bot.main as main
from crypto_bot.regime import regime_classifier


def test_ml_loader_logs_error(caplog, monkeypatch):
    """When ML loader fails, log the error without trainer hints."""
    importlib.reload(main)
def test_missing_supabase_logs_hint(monkeypatch, caplog):
    """When ML is enabled but unavailable, log Supabase guidance."""
    importlib.reload(main)
    from crypto_bot.utils import ml_utils

    monkeypatch.setattr(ml_utils, "ML_AVAILABLE", False)
    caplog.set_level(logging.INFO, logger="bot")

    async def boom(_symbol):
        raise RuntimeError("boom")

    monkeypatch.setattr(regime_classifier, "load_regime_model", boom)
    caplog.set_level(logging.ERROR, logger="bot")

    with pytest.raises(main.MLUnavailableError):
        main._ensure_ml_if_needed({"ml_enabled": True})

    assert "Machine learning initialization failed: boom" in caplog.text
    assert "cointrader-trainer" not in caplog.text
    assert "SUPABASE_URL" in caplog.text
    assert "cointrader-trainer" in caplog.text

