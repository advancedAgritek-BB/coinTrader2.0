from __future__ import annotations

from typing import Optional
import asyncio

from .telegram import send_message, send_message_sync


class Notifier:
    """Simple wrapper around :func:`send_message` for easy mocking."""

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id

    def notify(self, text: str) -> Optional[str]:
        """Send ``text`` via Telegram and return an error string if any."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(send_message(self.token, self.chat_id, text))
                return None
            except Exception as exc:  # pragma: no cover - network
                return str(exc)
        else:
            loop.create_task(send_message(self.token, self.chat_id, text))
            return None
            send_message_sync(self.token, self.chat_id, text)
            return None
        except Exception as err:  # pragma: no cover - network
            return str(err)

    async def notify_async(self, text: str) -> Optional[str]:
        """Asynchronously send ``text`` via Telegram."""
        try:
            await send_message(self.token, self.chat_id, text)
            return None
        except Exception as err:  # pragma: no cover - network
            return str(err)

