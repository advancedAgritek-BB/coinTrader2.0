from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Any
import asyncio
import inspect
import threading
import os
import time

from telegram import Bot

from .logger import LOG_DIR, setup_logger
from pathlib import Path


logger = setup_logger(__name__, LOG_DIR / "bot.log")

# Store allowed Telegram admin IDs parsed from the environment or configuration
_admin_ids: set[str] = set()


def set_admin_ids(admins: Iterable[str] | str | Any | None) -> None:
    """Configure allowed Telegram admin chat IDs."""
    global _admin_ids
    if admins is None:
        admins = os.getenv("TELE_CHAT_ADMINS", "")
    if isinstance(admins, str):
        parts = [a.strip() for a in admins.split(",") if a.strip()]
    elif isinstance(admins, Iterable):
        parts = [str(a).strip() for a in admins if str(a).strip()]
    else:
        parts = [str(admins).strip()] if str(admins).strip() else []
    _admin_ids = set(parts)


def is_admin(chat_id: str) -> bool:
    """Return ``True`` if ``chat_id`` is allowed to issue commands."""
    if not _admin_ids:
        return True
    return str(chat_id) in _admin_ids


set_admin_ids(None)


def send_message(token: str, chat_id: str, text: str) -> Optional[str]:
    """Send ``text`` to ``chat_id`` using ``token``.

    Returns ``None`` on success or an error string on failure. On timeout
    errors, the send is retried up to three times with exponential backoff.
    """

    bot = Bot(token)

    async def _send_async() -> Optional[str]:
        err: Optional[str] = None
        for attempt in range(3):
            try:
                await bot.send_message(chat_id=chat_id, text=text)
                return None
            except Exception as exc:  # pragma: no cover - network
                err = str(exc)
                if "timeout" in err.lower() and attempt < 2:
                    delay = 2 ** attempt
                    logger.warning("Telegram send timeout, retrying in %s seconds", delay)
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    "Failed to send message: %s. Verify your Telegram token "
                    "and chat ID and ensure the bot has started a chat.",
                    exc,
                )
                break
        return err

    def _send_sync() -> Optional[str]:
        err: Optional[str] = None
        for attempt in range(3):
            try:
                bot.send_message(chat_id=chat_id, text=text)
                return None
            except Exception as exc:  # pragma: no cover - network
                err = str(exc)
                if "timeout" in err.lower() and attempt < 2:
                    delay = 2 ** attempt
                    logger.warning(
                        "Telegram send timeout, retrying in %s seconds", delay
                    )
                    time.sleep(delay)
                    continue
                logger.error(
                    "Failed to send message: %s. Verify your Telegram token "
                    "and chat ID and ensure the bot has started a chat.",
                    exc,
                )
                break
        return err

    try:
        if inspect.iscoroutinefunction(bot.send_message):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # schedule and wait so that errors can be handled
                fut = asyncio.run_coroutine_threadsafe(_send_async(), loop)
                return fut.result()
            else:
                return asyncio.run(_send_async())
        else:
            return _send_sync()
    except Exception as e:  # pragma: no cover - network
        logger.error("Failed to send message: %s", e)
        return str(e)


@dataclass
class TelegramNotifier:
    """Simple notifier for sending Telegram messages."""

    token: str = ""
    chat_id: str = ""
    enabled: bool = True
    def __init__(
        self,
        enabled: bool = True,
        token: str = "",
        chat_id: str = "",
        admins: Iterable[str] | str | None = None,
    ) -> None:
        self.enabled = enabled
        self.token = token
        self.chat_id = chat_id
        if admins:
            set_admin_ids(admins)
        # internal flag set to True after a failed send
        self._disabled = False
        # lock to serialize send attempts
        self._lock = threading.Lock()

    def notify(self, text: str) -> Optional[str]:
        """Send ``text`` if notifications are enabled and credentials exist."""
        if self._disabled or not self.enabled or not self.token or not self.chat_id:
            return None

        with self._lock:
            if self._disabled:
                return None
            err: Optional[str] = None
            for attempt in range(3):
                err = send_message(self.token, self.chat_id, text)
                if err is None:
                    return None
                if "timeout" in err.lower() and attempt < 2:
                    delay = 2 ** attempt
                    logger.warning(
                        "Telegram send timeout, retrying in %s seconds", delay
                    )
                    time.sleep(delay)
                    continue
                self._disabled = True
                logger.error(
                    "Disabling Telegram notifications due to send failure: %s",
                    err,
                )
                break
            return err

    @classmethod
    def from_config(cls, config: dict) -> "TelegramNotifier":
        """Create a notifier from a configuration dictionary."""
        admins = config.get("chat_admins") or config.get("admins")
        notifier = cls(
            token=config.get("token", ""),
            chat_id=config.get("chat_id", ""),
            enabled=config.get("enabled", True),
            admins=admins,
        )
        return notifier


def send_test_message(token: str, chat_id: str, text: str = "Test message") -> bool:
    """Send a short test message to verify Telegram configuration."""
    if not token or not chat_id:
        return False
    err = send_message(token, chat_id, text)
    return err is None
