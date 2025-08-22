import logging
import sys
import types

import crypto_bot.utils.ml_utils as ml_utils
from crypto_bot.utils.ml_utils import ML_AVAILABLE


def test_ml_available_is_bool():
    assert isinstance(ML_AVAILABLE, bool)


def test_no_warning_when_models_available(monkeypatch, caplog):
    def fake_init():
        ml_utils.ML_AVAILABLE = True
        return True, ""

    monkeypatch.setattr(ml_utils, "init_ml_components", fake_init)
    monkeypatch.setattr(ml_utils, "is_ml_available", lambda: (True, ""))
    monkeypatch.setattr(ml_utils, "_ml_checked", False)
    monkeypatch.setattr(
        ml_utils, "_LOGGER_ONCE", {k: False for k in ml_utils._LOGGER_ONCE}
    )

    monkeypatch.setitem(sys.modules, "coinTrader_Trainer", types.ModuleType("coinTrader_Trainer"))
    monkeypatch.setitem(
        sys.modules,
        "coinTrader_Trainer.ml_trainer",
        types.ModuleType("coinTrader_Trainer.ml_trainer"),
    )

    caplog.set_level(logging.INFO)
    assert ml_utils.init_ml_or_warn() is True
    assert "Machine learning disabled" not in caplog.text
    caplog.clear()
    ml_utils.warn_ml_unavailable_once()
    assert "Machine learning disabled" not in caplog.text


def test_warn_ml_unavailable_logs_reason(monkeypatch, caplog):
    monkeypatch.setattr(
        ml_utils,
        "is_ml_available",
        lambda: (False, "Missing required ML packages: sklearn, joblib"),
    )
    monkeypatch.setattr(ml_utils, "_LOGGER_ONCE", {k: False for k in ml_utils._LOGGER_ONCE})
    caplog.set_level(logging.INFO)
    ml_utils.warn_ml_unavailable_once()
    assert (
        "Machine learning disabled: Missing required ML packages: sklearn, joblib"
        in caplog.text
    )
    caplog.clear()
    ml_utils.warn_ml_unavailable_once()
    assert caplog.text == ""
