from __future__ import annotations

import asyncio
from typing import Sequence, Any

from .utils.logger import setup_logger
from .utils.telegram import TelegramNotifier

logger = setup_logger(__name__, "crypto_bot/logs/telegram_ctl.log")


async def _maybe_call(func: Any) -> Any:
    """Call ``func`` which may be sync or async."""
    if asyncio.iscoroutinefunction(func):
        return await func()
    return await asyncio.to_thread(func)


async def status_loop(
    controller: Any,
    admins: Sequence[TelegramNotifier],
    update_interval: float = 60.0,
) -> None:
    """Periodically send status updates using ``controller``."""
    while True:
        try:
            status = await _maybe_call(controller.get_status)
            positions = await _maybe_call(controller.list_positions)
            lines = [str(status)]
            if positions:
                if isinstance(positions, str):
                    lines.append(positions)
                else:
                    lines.extend(str(p) for p in positions)
            message = "\n".join(lines)
            for admin in admins:
                admin.notify(message)
        except Exception as exc:  # pragma: no cover - logging only
            logger.error("Status update failed: %s", exc)
        await asyncio.sleep(update_interval)


def start(
    controller: Any,
    admins: Sequence[TelegramNotifier],
    update_interval: float = 60.0,
    enabled: bool = True,
) -> asyncio.Task | None:
    """Return background task sending periodic updates when ``enabled``."""
    if not enabled:
        return None
    return asyncio.create_task(status_loop(controller, admins, update_interval))
