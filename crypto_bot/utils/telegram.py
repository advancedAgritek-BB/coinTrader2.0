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

    Returns ``None`` on success or an error string on failure.
    """
    try:
        bot = Bot(token)

        async def _send() -> None:
            try:
                await bot.send_message(chat_id=chat_id, text=text)
            except Exception as exc:  # pragma: no cover - network
                logger.error(
                    "Failed to send message: %s. Verify your Telegram token "
                    "and chat ID and ensure the bot has started a chat.",
                    exc,
                )

        if inspect.iscoroutinefunction(bot.send_message):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                loop.create_task(_send())
            else:
                asyncio.run(_send())
        else:
            bot.send_message(chat_id=chat_id, text=text)
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
        message_interval: float = 1.0,
        max_per_minute: int = 20,
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
        # rate limiting
        self.message_interval = message_interval
        self.max_per_minute = max_per_minute
        self._last_sent = 0.0
        self._recent_sends: list[float] = []

    def notify(self, text: str) -> Optional[str]:
        """Send ``text`` if notifications are enabled and credentials exist."""
        if self._disabled or not self.enabled or not self.token or not self.chat_id:
            return None

        with self._lock:
            if self._disabled:
                return None

            now = time.time()
            self._recent_sends = [t for t in self._recent_sends if now - t < 60]

            delay = max(0.0, self.message_interval - (now - self._last_sent))
            if self._recent_sends and len(self._recent_sends) >= self.max_per_minute:
                oldest = self._recent_sends[0]
                delay = max(delay, 60 - (now - oldest))

            if delay > 0:
                time.sleep(delay)
                now = time.time()

            err = send_message(self.token, self.chat_id, text)
            if err is not None:
                self._disabled = True
                logger.error(
                    "Disabling Telegram notifications due to send failure: %s",
                    err,
                )
            else:
                self._last_sent = now
                self._recent_sends.append(now)
            return err

    async def notify_async(self, text: str) -> Optional[str]:
        """Asynchronous variant of :meth:`notify` using ``asyncio.sleep``."""
        if self._disabled or not self.enabled or not self.token or not self.chat_id:
            return None

        if not hasattr(self, "_async_lock"):
            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            if self._disabled:
                return None

            now = time.time()
            self._recent_sends = [t for t in self._recent_sends if now - t < 60]

            delay = max(0.0, self.message_interval - (now - self._last_sent))
            if self._recent_sends and len(self._recent_sends) >= self.max_per_minute:
                oldest = self._recent_sends[0]
                delay = max(delay, 60 - (now - oldest))

            if delay > 0:
                await asyncio.sleep(delay)
                now = time.time()

            err = send_message(self.token, self.chat_id, text)
            if err is not None:
                self._disabled = True
                logger.error(
                    "Disabling Telegram notifications due to send failure: %s",
                    err,
                )
            else:
                self._last_sent = now
                self._recent_sends.append(now)
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
            message_interval=config.get("message_interval", 1.0),
            max_per_minute=config.get("max_messages_per_minute", 20),
        )
        return notifier


def send_test_message(token: str, chat_id: str, text: str = "Test message") -> bool:
    """Send a short test message to verify Telegram configuration."""
    if not token or not chat_id:
        return False
    err = send_message(token, chat_id, text)
    return err is None
