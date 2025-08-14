from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class EvalGate:
    def __init__(self):
        self._busy = False

    def is_busy(self) -> bool:
        return self._busy

    @contextmanager
    def hold(self, note: str = ""):
        self._busy = True
        try:
            yield
        finally:
            self._busy = False
            if note:
                logger.debug(f"[EvalGate] released: {note}")


eval_gate = EvalGate()
