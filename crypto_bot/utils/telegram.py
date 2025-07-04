from telegram import Bot
from typing import Optional
import inspect
import asyncio

from .logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/bot.log")


def send_message(token: str, chat_id: str, text: str) -> Optional[str]:
    try:
        bot = Bot(token)
        if inspect.iscoroutinefunction(bot.send_message):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                loop.create_task(bot.send_message(chat_id=chat_id, text=text))
            else:
                asyncio.run(bot.send_message(chat_id=chat_id, text=text))
        else:
            bot.send_message(chat_id=chat_id, text=text)
        return None
    except Exception as e:
        logger.error("Failed to send message: %s", e)
        return str(e)
