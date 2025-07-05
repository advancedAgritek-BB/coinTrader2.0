from dataclasses import dataclass
from telegram import Bot
from typing import Optional
from dataclasses import dataclass
from telegram import Bot
import inspect
import asyncio

from .logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


class TelegramNotifier:
    """Simple wrapper for sending Telegram notifications."""

    def __init__(self, token: str = "", chat_id: str = "", enabled: bool = True) -> None:
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled and bool(token and chat_id)

    @classmethod
    def from_config(cls, config: dict) -> "TelegramNotifier":
        """Instantiate from a configuration dictionary."""
        return cls(
            token=config.get("token", ""),
            chat_id=config.get("chat_id", ""),
            enabled=config.get("enabled", True),
        )

    def notify(self, text: str) -> Optional[str]:
        """Send a message if enabled and credentials are provided."""
        if not self.enabled:
            return None
        try:
            bot = Bot(self.token)

            async def _send() -> None:
                try:
                    await bot.send_message(chat_id=self.chat_id, text=text)
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
                bot.send_message(chat_id=self.chat_id, text=text)
            return None
        except Exception as e:  # pragma: no cover - network
            logger.error("Failed to send message: %s", e)
            return str(e)


def send_message(token: str, chat_id: str, text: str) -> Optional[str]:
    """Backward compatible helper for sending a Telegram message."""
    notifier = TelegramNotifier(token, chat_id)
    return notifier.notify(text)
@dataclass
class TelegramNotifier:
    """Lightweight notifier for sending Telegram messages."""
    """Simple notifier for sending Telegram messages."""

    token: str
    chat_id: str

    def notify(self, text: str) -> Optional[str]:
        """Send a Telegram message using the stored credentials."""
        """Send ``text`` as a Telegram message."""
    def send(self, text: str) -> Optional[str]:
        """Send ``text`` via Telegram."""
        return send_message(self.token, self.chat_id, text)


def send_message(token: str, chat_id: str, text: str) -> Optional[str]:
    try:
        bot = Bot(token)

        async def _send() -> None:
            try:
                await bot.send_message(chat_id=chat_id, text=text)
            except Exception as exc:
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
    except Exception as e:
        logger.error("Failed to send message: %s", e)
        return str(e)


class TelegramNotifier:
    """Helper class for sending Telegram notifications."""

    def __init__(self, enabled: bool, token: str, chat_id: str) -> None:
        self.enabled = enabled
        self.token = token
        self.chat_id = chat_id

    def notify(self, text: str) -> Optional[str]:
        """Send a message if enabled."""
        if not self.enabled:
            return None
        return send_message(self.token, self.chat_id, text)

    @classmethod
    def from_config(cls, cfg: dict) -> "TelegramNotifier":
        """Create notifier from config dict."""
        section = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}
        enabled = bool(section.get("enabled", False))
        token = section.get("token", "")
        chat_id = section.get("chat_id", "")
        return cls(enabled, token, chat_id)
def send_test_message(token: str, chat_id: str, text: str = "Test message") -> bool:
    """Send a short test message to verify Telegram configuration.

    Returns ``True`` if the message was sent successfully, otherwise ``False``.
    """

    if not token or not chat_id:
        return False
    err = send_message(token, chat_id, text)
    return err is None
