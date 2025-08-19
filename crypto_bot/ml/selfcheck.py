import importlib
import logging
import os

from crypto_bot.ml.model_loader import get_supabase_key

_logged = False

_REQUIRED_PACKAGES = ("sklearn", "joblib", "ta")


def log_ml_status_once() -> None:
    """Log Supabase ML environment status once."""
    global _logged
    if _logged:
        return
    _logged = True
    log = logging.getLogger("crypto_bot.ml")
    pkgs_ok = all(
        importlib.util.find_spec(name) is not None for name in _REQUIRED_PACKAGES
    )
    url_ok = bool(os.getenv("SUPABASE_URL"))
    key_ok = bool(get_supabase_key())
    log.info(
        "ML status: packages=%s supabase_url=%s SUPABASE_KEY_present=%s",
        pkgs_ok,
        url_ok,
        key_ok,
    )
