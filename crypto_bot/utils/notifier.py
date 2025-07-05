from __future__ import annotations

from typing import Optional

from .telegram import send_message


class Notifier:
    """Simple wrapper around :func:`send_message` for easy mocking."""

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id

    def notify(self, text: str) -> Optional[str]:
        """Send ``text`` via Telegram and return an error string if any."""
        return send_message(self.token, self.chat_id, text)

