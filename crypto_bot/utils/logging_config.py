import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .logger import LOG_DIR


class LastLineBuffer(logging.Handler):
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self.last = ""

    def emit(self, record):
        try:
            self.last = record.getMessage()
        except Exception:
            pass


def setup_logging(log_path: str | os.PathLike = LOG_DIR / "bot.log", level=logging.INFO):
    log_path = Path(log_path)
    os.makedirs(log_path.parent, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    # File handler for everything
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s - %(message)s"))
    root.addHandler(fh)

    # Silence noisy libs
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    # Attach last-line buffer (not a console printer)
    last = LastLineBuffer(level=logging.INFO)
    root.addHandler(last)
    return last
