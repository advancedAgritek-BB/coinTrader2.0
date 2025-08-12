from __future__ import annotations

from typing import Optional
import asyncio

from .telegram import send_message, send_message_sync


class Notifier:
    """Simple wrapper around :func:`send_message` for easy mocking."""

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id

    async def notify_async(
        self, text: str, retries: int = 3, timeout: float = 5.0
    ) -> Optional[str]:
        """Asynchronously send ``text`` via Telegram.

        Parameters
        ----------
        text:
            Message to send.
        retries:
            Number of attempts if a timeout occurs. Defaults to ``3``.
        timeout:
            Maximum time in seconds to wait per attempt. Defaults to ``5``.

        Returns
        -------
        Optional[str]
            ``None`` on success or an error string on failure.
        """

        for attempt in range(1, retries + 1):
            try:
                return await asyncio.wait_for(
                    send_message(self.token, self.chat_id, text),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                if attempt == retries:
                    return f"Timed out after {retries} attempts"
            except ValueError as exc:
                return str(exc)
            except Exception as exc:  # pragma: no cover - unexpected errors
                return str(exc)
        return f"Timed out after {retries} attempts"

    def notify(
        self, text: str, retries: int = 3, timeout: float = 5.0
    ) -> Optional[str]:
        """Send ``text`` via Telegram and return an error string if any.

        If called within an active event loop, the message is sent synchronously
        to avoid ``asyncio.run`` errors. Otherwise the asynchronous implementation
        is executed using :func:`asyncio.run`.
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            try:
                return asyncio.run(
                    self.notify_async(text, retries=retries, timeout=timeout)
                )
            except ValueError as exc:
                return str(exc)

        # Running inside an event loop – fall back to synchronous sending
        for attempt in range(1, retries + 1):
            try:
                return send_message_sync(self.token, self.chat_id, text)
            except TimeoutError:
                if attempt == retries:
                    return f"Timed out after {retries} attempts"
            except ValueError as exc:
                return str(exc)
            except Exception as exc:  # pragma: no cover - unexpected errors
                return str(exc)
        return f"Timed out after {retries} attempts"

