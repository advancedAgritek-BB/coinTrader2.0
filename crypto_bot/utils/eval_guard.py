from contextlib import contextmanager
import logging
import time

logger = logging.getLogger(__name__)


class EvalGate:
    """Simple gate with TTL auto-release to prevent deadlocks."""

    def __init__(self, ttl_sec: float = 30.0):
        self._busy = False
        self._since = 0.0
        self._ttl = ttl_sec

    def is_busy(self) -> bool:
        if self._busy and (time.monotonic() - self._since) > self._ttl:
            logger.warning("Gate held >%ss; forcing release", self._ttl)
            self._busy = False
            self._since = 0.0
        return self._busy

    @contextmanager
    def hold(self, note: str = ""):
        self._busy = True
        self._since = time.monotonic()
        try:
            yield
        finally:
            self._busy = False
            self._since = 0.0
            if note:
                logger.debug(f"[EvalGate] released: {note}")


eval_gate = EvalGate()
