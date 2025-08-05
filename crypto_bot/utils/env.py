import os


def env_or_prompt(name: str, prompt: str) -> str:
    """Return the value of an environment variable or prompt the user."""
    return os.getenv(name) or input(prompt)
import sys

from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "env.log")


def env_or_prompt(name: str, prompt: str) -> str:
    """Return ``name`` from ``os.environ`` or prompt the user.

    In non-interactive environments (e.g. during testing) where prompting is
    not possible, an empty string is returned and a warning is logged.
    The obtained value is stored back into ``os.environ`` for reuse.
    """

    value = os.getenv(name)
    if value:
        return value

    if os.environ.get("PYTEST_CURRENT_TEST") or not sys.stdin or not sys.stdin.isatty():
        logger.warning(
            "Environment variable %s not set and input is non-interactive", name
        )
        return ""

    logger.info("Prompting user for %s", name)
    value = input(prompt)
    os.environ[name] = value
    return value


__all__ = ["env_or_prompt"]
