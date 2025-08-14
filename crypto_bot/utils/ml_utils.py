"""Utility helpers for optional machine learning dependencies."""
import importlib
import base64
import json
import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_REQUIRED_PACKAGES: Iterable[str] = ("sklearn", "joblib", "ta")

_ml_checked = False
_LOGGER_ONCE = {
    "ml_unavailable": False,
    "missing_supabase_creds": False,
    "anon_key_role": False,
}


def warn_ml_unavailable_once() -> None:
    """Log a one-time notice when ML components are missing."""
    if not _LOGGER_ONCE["ml_unavailable"]:
        logger.info(
            "Machine learning model not found; running without ML features."
        )
        _LOGGER_ONCE["ml_unavailable"] = True


def _check_packages(pkgs: Iterable[str]) -> bool:
    """Return ``True`` if all packages in ``pkgs`` can be imported."""
    return all(importlib.util.find_spec(name) is not None for name in pkgs)


def _get_supabase_creds() -> tuple[str | None, str | None]:
    """Return Supabase URL and key from canonical or legacy env names."""
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_API_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
    )
    logger.debug(
        "Supabase configured: url=%s key_len=%d",
        bool(url),
        len(key or ""),
    )
    return url, key


def _warn_if_anon_key(key: str) -> None:
    """Warn once if ``key`` appears to be an anon key."""
    if _LOGGER_ONCE["anon_key_role"]:
        return
    try:  # pragma: no cover - best effort
        payload_b64 = key.split(".")[1]
        padding = "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
        role = payload.get("role")
        if role and role != "service_role":
            logger.warning("Supabase key has non-service role: %s", role)
            _LOGGER_ONCE["anon_key_role"] = True
    except Exception:
        # Ignore malformed keys or decoding issues
        pass


def is_ml_available() -> bool:
    """Return ``True`` if optional ML dependencies and model are available."""
    global _ml_checked, ML_AVAILABLE
    if _ml_checked:
        return ML_AVAILABLE
    _ml_checked = True

    try:
        try:  # optional training utilities
            import cointrader_trainer  # noqa: F401
        except ImportError:
            logger.info("ML disabled: cointrader-trainer not installed")
            ML_AVAILABLE = False
            return False

        if not _check_packages(_REQUIRED_PACKAGES):
            raise ImportError("Missing required ML packages")

        url, key = _get_supabase_creds()
        if not url or not key:
            if not _LOGGER_ONCE["missing_supabase_creds"]:
                logger.info(
                    "ML unavailable: Missing Supabase credentials (url=%s, key_present=%s)",
                    bool(url),
                    bool(key),
                )
                _LOGGER_ONCE["missing_supabase_creds"] = True
            return False

        _warn_if_anon_key(key)

        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "meta_selector_lgbm.txt"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        ML_AVAILABLE = True
        return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("ML unavailable: %s", exc)
        ML_AVAILABLE = False
        return False


ML_AVAILABLE = False


def init_ml_components() -> bool:
    """Initialize ML components and update :data:`ML_AVAILABLE`."""
    global ML_AVAILABLE
    ML_AVAILABLE = is_ml_available()
    return ML_AVAILABLE


__all__ = ["is_ml_available", "ML_AVAILABLE", "warn_ml_unavailable_once", "init_ml_components"]
