import logging
import os

_logged = False


def log_ml_status_once() -> None:
    """Log ML package and Supabase environment status once."""
    global _logged
    if _logged:
        return
    _logged = True
    log = logging.getLogger("crypto_bot.ml")
    try:  # pragma: no cover - optional dependency
        import cointrader_trainer  # noqa: F401
        pkg = True
    except ImportError:  # pragma: no cover - package missing
        pkg = False
    url_ok = bool(os.getenv("SUPABASE_URL"))
    key_ok = bool(
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_API_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )
    log.info(
        "ML status: package=%s supabase_url=%s key_present=%s", pkg, url_ok, key_ok
    )
