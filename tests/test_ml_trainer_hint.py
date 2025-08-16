import importlib
import logging

import crypto_bot.main as main


def test_missing_supabase_logs_hint(monkeypatch, caplog):
    """When ML is enabled but unavailable, log Supabase guidance."""
    importlib.reload(main)
    from crypto_bot.utils import ml_utils

    monkeypatch.setattr(ml_utils, "ML_AVAILABLE", False)
    caplog.set_level(logging.INFO, logger="bot")

    main._ensure_ml_if_needed({"ml_enabled": True})

    assert "SUPABASE_URL" in caplog.text
    assert "cointrader-trainer" in caplog.text

