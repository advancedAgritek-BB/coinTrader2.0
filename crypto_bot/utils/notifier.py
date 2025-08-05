from __future__ import annotations

import asyncio
from typing import Optional

from .telegram import send_message


class Notifier:
    """Thin wrapper around telegram helpers for ease of mocking."""

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id

    def notify(self, text: str) -> Optional[str]:
        """Synchronously send ``text`` via Telegram.

        Returns ``None`` on success or an error string on failure.
        """
        try:
            asyncio.run(send_message(self.token, self.chat_id, text))
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
