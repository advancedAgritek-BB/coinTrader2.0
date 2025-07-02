from telegram import Bot
from typing import Optional
import inspect
import asyncio


def send_message(token: str, chat_id: str, text: str) -> Optional[str]:
    try:
        bot = Bot(token)
        if inspect.iscoroutinefunction(bot.send_message):
            asyncio.run(bot.send_message(chat_id=chat_id, text=text))
        else:
            bot.send_message(chat_id=chat_id, text=text)
        return None
    except Exception as e:
        return str(e)
