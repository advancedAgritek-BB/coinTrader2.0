from telegram import Bot
from typing import Optional
import inspect
import asyncio
import threading


def send_message(token: str, chat_id: str, text: str) -> Optional[str]:
    try:
        bot = Bot(token)
        if inspect.iscoroutinefunction(bot.send_message):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                exc: list[Exception] = []

                def run():
                    try:
                        asyncio.run(bot.send_message(chat_id=chat_id, text=text))
                    except Exception as e:  # pragma: no cover - propagation tested via return
                        exc.append(e)

                thread = threading.Thread(target=run)
                thread.start()
                thread.join()
                if exc:
                    raise exc[0]
            if loop is not None:
                loop.create_task(bot.send_message(chat_id=chat_id, text=text))
            else:
                asyncio.run(bot.send_message(chat_id=chat_id, text=text))
        else:
            bot.send_message(chat_id=chat_id, text=text)
        return None
    except Exception as e:
        return str(e)
