from __future__ import annotations

"""Utility class for sending Telegram notifications."""

from dataclasses import dataclass
from typing import ClassVar, Optional

from .telegram import send_message


@dataclass
class TelegramNotifier:
    """Send Telegram notifications with optional global disable."""

    token: str
    chat_id: str

    enabled: ClassVar[bool] = True

    @classmethod
    def configure(cls, enabled: bool) -> None:
        """Globally enable or disable notifications."""
        cls.enabled = enabled

    def notify(self, text: str) -> Optional[str]:
        """Send ``text`` to the configured chat if enabled."""
        if not self.enabled or not self.token or not self.chat_id:
            return None
        return send_message(self.token, self.chat_id, text)

    @classmethod
    def send(cls, token: str, chat_id: str, text: str) -> Optional[str]:
        """Backwards compatible helper to send a single message."""
        if not cls.enabled or not token or not chat_id:
            return None
        return send_message(token, chat_id, text)
