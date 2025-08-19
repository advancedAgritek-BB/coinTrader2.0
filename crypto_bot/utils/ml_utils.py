"""Utility helpers for optional machine learning dependencies."""
import base64
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable

from crypto_bot.ml.model_loader import _supabase_key

logger = logging.getLogger(__name__)

_REQUIRED_PACKAGES: Iterable[str] = ("sklearn", "joblib", "ta")

_ml_checked = False
_LOGGER_ONCE = {
    "ml_unavailable": False,
    "missing_supabase_creds": False,
    "service_key_role": False,
    "supabase_connection": False,
}


def warn_ml_unavailable_once() -> None:
    """Log a one-time notice when ML components are missing."""
    if not _LOGGER_ONCE["ml_unavailable"]:
        logger.info(
            "Machine learning model not found; running without ML features."
        )
        _LOGGER_ONCE["ml_unavailable"] = True


def _check_packages(pkgs: Iterable[str]) -> list[str]:
    """Return a list of packages from ``pkgs`` that cannot be imported."""
    missing = []
    for name in pkgs:
        try:  # pragma: no cover - optional dependency import
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def _get_supabase_creds() -> tuple[str | None, str | None]:
    """Return Supabase URL and key using shared helper for key lookup."""
    url = os.getenv("SUPABASE_URL")
    key = _supabase_key()
    logger.debug(
        "Supabase configured: SUPABASE_URL=%s SUPABASE_KEY_len=%d",
        bool(url),
        len(key or ""),
    )
    return url, key


def _warn_if_service_key(key: str) -> None:
    """Warn once if ``key`` appears to be a service role key."""
    if _LOGGER_ONCE["service_key_role"]:
        return
    try:  # pragma: no cover - best effort
        payload_b64 = key.split(".")[1]
        padding = "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
        role = payload.get("role")
        if role == "service_role":
            logger.warning("Supabase key uses service role; prefer anon key for runtime")
            _LOGGER_ONCE["service_key_role"] = True
    except Exception:
        # Ignore malformed keys or decoding issues
        pass


def _test_supabase_connection() -> None:
    """Attempt a simple Supabase query to confirm connectivity."""
    if _LOGGER_ONCE["supabase_connection"]:
        return
    try:  # pragma: no cover - supabase optional
        from supabase import create_client  # type: ignore

        client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        client.table("models").select("*").limit(1).execute()
        logger.debug("Supabase connection succeeded")
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("Supabase connection failed: %s", exc)
    _LOGGER_ONCE["supabase_connection"] = True


def is_ml_available() -> tuple[bool, str]:
    """Return ML availability along with a descriptive reason if unavailable."""
    global _ml_checked, ML_AVAILABLE, ML_UNAVAILABLE_REASON
    if _ml_checked:
        return ML_AVAILABLE, ML_UNAVAILABLE_REASON
    _ml_checked = True

    try:
        missing = _check_packages(_REQUIRED_PACKAGES)
        if missing:
            reason = "Missing required ML packages: " + ", ".join(missing)
            logger.info("ML unavailable: %s", reason)
            ML_AVAILABLE = False
            ML_UNAVAILABLE_REASON = reason
            return ML_AVAILABLE, reason

        try:  # cointrader-trainer is optional and only used for training
            import cointrader_trainer  # noqa: F401
        except ImportError:
            logger.debug(
                "cointrader-trainer not installed; proceeding with runtime model download",
            )

        url, key = _get_supabase_creds()
        if not url or not key:
            if not _LOGGER_ONCE["missing_supabase_creds"]:
                logger.info(
                    "ML unavailable: Missing Supabase credentials (SUPABASE_URL=%s, SUPABASE_KEY_present=%s)",
                    bool(url),
                    bool(key),
                )
                _LOGGER_ONCE["missing_supabase_creds"] = True
            reason = "Missing Supabase credentials"
            ML_AVAILABLE = False
            ML_UNAVAILABLE_REASON = reason
            return ML_AVAILABLE, reason

        _warn_if_service_key(key)

        _test_supabase_connection()

        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "meta_selector_lgbm.txt"
        )
        if not model_path.exists():
            reason = f"Model file not found: {model_path}"
            logger.info("ML unavailable: %s", reason)
            ML_AVAILABLE = False
            ML_UNAVAILABLE_REASON = reason
            return ML_AVAILABLE, reason

        ML_AVAILABLE = True
        ML_UNAVAILABLE_REASON = ""
        return ML_AVAILABLE, ""
    except Exception as exc:  # pragma: no cover - best effort
        reason = str(exc)
        logger.error("ML unavailable: %s", reason)
        ML_AVAILABLE = False
        ML_UNAVAILABLE_REASON = reason
        return ML_AVAILABLE, reason
ML_AVAILABLE = False
ML_UNAVAILABLE_REASON = ""


def init_ml_components() -> tuple[bool, str]:
    """Initialize ML components and update availability globals."""
    global ML_AVAILABLE, ML_UNAVAILABLE_REASON
    ML_AVAILABLE, ML_UNAVAILABLE_REASON = is_ml_available()
    return ML_AVAILABLE, ML_UNAVAILABLE_REASON


__all__ = [
    "is_ml_available",
    "ML_AVAILABLE",
    "ML_UNAVAILABLE_REASON",
    "warn_ml_unavailable_once",
    "init_ml_components",
]
