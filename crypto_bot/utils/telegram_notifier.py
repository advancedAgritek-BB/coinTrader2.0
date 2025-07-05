from __future__ import annotations

"""Utility class for sending Telegram notifications with optional global disable."""

"""Simple Telegram notifier."""

from dataclasses import dataclass
from typing import Optional

from .telegram import send_message


class TelegramNotifier:
    """Simple wrapper around :func:`send_message` with a global enable flag."""

    enabled: bool = True

    @classmethod
    def configure(cls, enabled: bool) -> None:
        """Enable or disable notifications globally."""
        cls.enabled = enabled

    @classmethod
    def notify(cls, token: str, chat_id: str, text: str) -> Optional[str]:
        """Send a message if notifications are enabled and credentials are provided."""
        if not cls.enabled or not token or not chat_id:
            return None
        return send_message(token, chat_id, text)
@dataclass
class TelegramNotifier:
    """Send Telegram notifications using :func:`send_message`."""

    token: str
    chat_id: str

    def notify(self, text: str) -> Optional[str]:
        """Send ``text`` to the configured chat.

        Uses :func:`send_message` so it works in running event loops.
        """
        return send_message(self.token, self.chat_id, text)
