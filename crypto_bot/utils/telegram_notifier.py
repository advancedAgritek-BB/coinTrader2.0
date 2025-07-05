from __future__ import annotations

"""Simple Telegram notifier."""

from dataclasses import dataclass
from typing import Optional

from .telegram import send_message


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
