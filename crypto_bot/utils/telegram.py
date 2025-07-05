from dataclasses import dataclass
from telegram import Bot
from typing import Optional
import inspect
import asyncio

from .logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


@dataclass
class TelegramNotifier:
    """Simple notifier for sending Telegram messages."""

    token: str
    chat_id: str

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
