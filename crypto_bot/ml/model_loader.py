import logging
import os

logger = logging.getLogger(__name__)

_SUPABASE_KEY_VARS = (
    "SUPABASE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_API_KEY",
    "SUPABASE_ANON_KEY",
)

def get_supabase_key() -> str | None:
    """Return Supabase key preferring ``SUPABASE_KEY`` with legacy fallbacks."""
    for name in _SUPABASE_KEY_VARS:
        key = os.getenv(name)
        if key:
            if name != "SUPABASE_KEY":
                logger.debug("Using legacy Supabase key env %s", name)
            return key
    return None

__all__ = ["get_supabase_key"]
