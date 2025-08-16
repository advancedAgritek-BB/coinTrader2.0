import importlib
import logging

import crypto_bot.main as main


def test_missing_trainer_logs_install_hint(caplog):
    """When ML is enabled but cointrader-trainer is missing, log hint."""
    importlib.reload(main)
    main._TRAINER_AVAILABLE = False
    caplog.set_level(logging.INFO, logger="bot")

    main._ensure_ml_if_needed({"ml_enabled": True})

    assert "pip install cointrader-trainer" in caplog.text

