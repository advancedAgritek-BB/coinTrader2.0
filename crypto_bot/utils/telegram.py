from __future__ import annotations

import asyncio
import inspect
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from telegram import Bot
from telegram.error import RetryAfter

from .logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "bot.log")

# Store allowed Telegram admin IDs parsed from the environment or configuration
_admin_ids: set[str] = set()

# timestamp of the last successful message send
_last_send: float = 0.0
# lock to serialize rate-limited send attempts
_send_lock = threading.Lock()


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

    Returns ``None`` on success or an error string on failure.
    """
    try:
        bot = Bot(token)

        if inspect.iscoroutinefunction(bot.send_message):

            async def _send() -> None:
                global _last_send
                with _send_lock:
                    wait = 0.0 if _last_send <= 0 else 1.0 - (time.time() - _last_send)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    try:
                        await bot.send_message(chat_id=chat_id, text=text)
                    except RetryAfter as exc:
                        await asyncio.sleep(float(exc.retry_after))
                        await bot.send_message(chat_id=chat_id, text=text)
                    _last_send = time.time()

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                loop.create_task(_send())
            else:
                asyncio.run(_send())
        else:
            global _last_send
            with _send_lock:
                wait = 0.0 if _last_send <= 0 else 1.0 - (time.time() - _last_send)
                if wait > 0:
                    time.sleep(wait)
                try:
                    bot.send_message(chat_id=chat_id, text=text)
                except RetryAfter as exc:
                    time.sleep(float(exc.retry_after))
                    bot.send_message(chat_id=chat_id, text=text)
                _last_send = time.time()
        return None
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
            err = send_message(self.token, self.chat_id, text)
            if err is not None:
                self._disabled = True
                logger.error(
                    "Disabling Telegram notifications due to send failure: %s",
                    err,
                )
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
