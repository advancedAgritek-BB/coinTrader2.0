from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import asyncio
import inspect

from telegram import Bot

from .logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


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
    def __init__(self, enabled: bool = True, token: str = "", chat_id: str = "") -> None:
        self.enabled = enabled
        self.token = token
        self.chat_id = chat_id
        # internal flag set to True after a failed send
        self._disabled = False

    def notify(self, text: str) -> Optional[str]:
        """Send ``text`` if notifications are enabled and credentials exist."""
        if self._disabled or not self.enabled or not self.token or not self.chat_id:
            return None

        err = send_message(self.token, self.chat_id, text)
        if err is not None:
            self._disabled = True
            logger.error("Disabling Telegram notifications due to send failure: %s", err)
        return err

    @classmethod
    def from_config(cls, config: dict) -> "TelegramNotifier":
        """Create a notifier from a configuration dictionary."""
        return cls(
            token=config.get("token", ""),
            chat_id=config.get("chat_id", ""),
            enabled=config.get("enabled", True),
        )


def send_test_message(token: str, chat_id: str, text: str = "Test message") -> bool:
    """Send a short test message to verify Telegram configuration."""
    if not token or not chat_id:
        return False
    err = send_message(token, chat_id, text)
    return err is None
