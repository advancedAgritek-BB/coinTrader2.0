from telegram import Bot
from typing import Optional
import inspect
import asyncio

from .logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


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


def send_test_message(token: str, chat_id: str, text: str = "Test message") -> bool:
    """Send a short test message to verify Telegram configuration.

    Returns ``True`` if the message was sent successfully, otherwise ``False``.
    """

    if not token or not chat_id:
        return False
    err = send_message(token, chat_id, text)
    return err is None
