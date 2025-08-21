import asyncio
import logging
from collections import Counter, defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


class TaskManager:
    """Manage asyncio tasks for the bot.

    Tasks can be registered as background tasks and grouped by name for
    later inspection or cancellation. Failures are tracked and optionally
    reported via a notifier.
    """

    def __init__(self, notify_threshold: int = 3) -> None:
        self._tasks: dict[str, set[asyncio.Task]] = defaultdict(set)
        self.failure_counts: Counter[str] = Counter()
        self._notifier: Optional[object] = None
        self.notify_threshold = notify_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register(self, task: asyncio.Task | None) -> asyncio.Task | None:
        """Register a background task and attach failure monitoring."""
        if not task:
            return None
        self._tasks["background"].add(task)
        task.add_done_callback(self._handle_completion)
        return task

    def add(self, task: asyncio.Task | None, group: str, remove_on_done: bool = False) -> None:
        """Track ``task`` under ``group``.

        If ``remove_on_done`` is True, the task is discarded from the group
        when it completes.
        """
        if not task:
            return
        self._tasks[group].add(task)
        if remove_on_done:
            task.add_done_callback(lambda t, g=group: self._tasks[g].discard(t))

    def prune(self, group: str) -> None:
        """Remove completed tasks from ``group``."""
        for task in list(self._tasks.get(group, [])):
            if task.done():
                self._tasks[group].discard(task)

    async def cancel_all(self) -> None:
        """Cancel and await all tracked tasks."""
        all_tasks = [t for tasks in self._tasks.values() for t in tasks]
        for t in all_tasks:
            t.cancel()
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        self._tasks.clear()

    def set_notifier(self, notifier: object) -> None:
        """Set a notifier used for reporting repeated task failures."""
        self._notifier = notifier

    @property
    def tasks(self) -> dict[str, set[asyncio.Task]]:
        """Expose tracked task groups for introspection/testing."""
        return self._tasks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _handle_completion(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        name = task.get_name()
        logger.error("Background task %s raised an exception", name, exc_info=exc)
        self.failure_counts[name] += 1
        if (
            self._notifier
            and self.failure_counts[name] >= self.notify_threshold
        ):
            try:
                self._notifier.notify(
                    f"Task {name} failed {self.failure_counts[name]} times: {exc}"
                )
            except Exception:
                logger.exception("Failed to send failure notification for %s", name)


