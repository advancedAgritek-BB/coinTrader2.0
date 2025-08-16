import logging
import os

_logged = False


def log_ml_status_once() -> None:
    """Log Supabase ML environment status once."""
    global _logged
    if _logged:
        return
    _logged = True
    log = logging.getLogger("crypto_bot.ml")
    url_ok = bool(os.getenv("SUPABASE_URL"))
    key_ok = bool(
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_API_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )
    log.info("ML status: supabase_url=%s key_present=%s", url_ok, key_ok)
