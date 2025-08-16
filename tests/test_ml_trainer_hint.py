import sys
import logging

from crypto_bot.utils import ml_utils


def test_trainer_hint_logged_when_trainer_missing(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    monkeypatch.setattr(ml_utils, "_check_packages", lambda _pkgs: ["pkg"])
    monkeypatch.delitem(sys.modules, "cointrader_trainer", raising=False)
    ml_utils._ml_checked = False
    ml_utils.ML_AVAILABLE = False

    ml_utils.is_ml_available()

    assert (
        "cointrader-trainer not installed; proceeding with runtime model download"
        in caplog.text
    )
