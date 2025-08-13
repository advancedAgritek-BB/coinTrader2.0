from __future__ import annotations

import asyncio
import httpx
from telegram.error import NetworkError

from crypto_bot.utils.logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "telegram_polling.log")


async def run_polling(bot) -> None:
    """Continuously poll Telegram ``bot`` with exponential backoff.

    Parameters
    ----------
    bot:
        Instance of :class:`telegram.Bot` or compatible object exposing
        ``get_updates``.
    """
    backoff = 2.0
    while True:
        try:
            await bot.get_updates(timeout=60, read_timeout=70, write_timeout=70)
            backoff = 2.0
        except (NetworkError, httpx.ReadError) as exc:
            logger.warning(
                "Telegram polling error: %s; retrying in %.1fs", exc, backoff
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
        except Exception as exc:  # pragma: no cover - unexpected
            logger.exception(
                "Telegram polling fatal error; continuing loop: %s", exc
            )
            await asyncio.sleep(5.0)

