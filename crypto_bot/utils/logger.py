import logging
import os
from pathlib import Path
from typing import Dict, Tuple

# Default directory for all log files used across the project. Users can
# override the location by setting the ``LOG_DIR`` environment variable.
# If not provided, logs are written under ``crypto_bot/logs`` within the
# repository.
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR = Path(os.environ.get("LOG_DIR", DEFAULT_LOG_DIR)).expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)


class ZeroScoreFilter(logging.Filter):
    """Filter that suppresses repeated zero-score "no signal" logs.

    Messages of the form "Signal for ... -> 0.0, none" are only emitted when the
    score or direction changes compared to the last logged state for a given
    symbol/timeframe pair.  This keeps log files from being flooded with
    identical "no signal" entries while still reporting meaningful transitions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_state: Dict[Tuple[str, str], Tuple[float, str]] = {}

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - logging
        # Only process standard signal messages
        if not record.args or not str(record.msg).startswith("Signal for"):
            return True

        try:
            symbol, timeframe, score, direction = record.args[:4]
        except Exception:  # pragma: no cover - defensiveness
            return True

        key = (str(symbol), str(timeframe))
        prev_score, prev_dir = self.last_state.get(key, (None, None))
        if score == 0.0 and direction == "none" and prev_score == 0.0 and prev_dir == "none":
            return False

        self.last_state[key] = (score, direction)
        return True


def setup_logger(name: str, log_file: Path | str, to_console: bool = True) -> logging.Logger:
    """Return a logger configured to write to ``log_file`` within ``LOG_DIR`` and optionally stdout.

    The directory ``LOG_DIR`` is created automatically when the logger is initialized.
    """

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler_exists = any(
        isinstance(h, logging.FileHandler)
        and Path(getattr(h, "baseFilename", "")) == log_file
        for h in logger.handlers
    )
    if not file_handler_exists:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if to_console and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if name == "symbol_filter" and not any(
        isinstance(f, ZeroScoreFilter) for f in logger.filters
    ):
        logger.addFilter(ZeroScoreFilter())

    return logger


# Shared logger for all indicator related modules
indicator_logger = setup_logger("indicators", LOG_DIR / "indicators.log")

# Shared logger for pipeline stages
pipeline_logger = setup_logger("pipeline", LOG_DIR / "pipeline.log")
