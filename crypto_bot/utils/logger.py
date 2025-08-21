import logging
import os
from pathlib import Path

# Default directory for all log files used across the project. Users can
# override the location by setting the ``LOG_DIR`` environment variable.
# If not provided, logs are written under ``crypto_bot/logs`` within the
# repository.
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR = Path(os.environ.get("LOG_DIR", DEFAULT_LOG_DIR)).expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)


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

    return logger


# Shared logger for all indicator related modules
indicator_logger = setup_logger("indicators", LOG_DIR / "indicators.log")

# Shared logger for pipeline stages
pipeline_logger = setup_logger("pipeline", LOG_DIR / "pipeline.log")
