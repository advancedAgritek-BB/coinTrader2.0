import logging
from pathlib import Path


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Return a logger configured to write to ``log_file`` and stdout."""

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler_exists = any(
        isinstance(h, logging.FileHandler)
        and Path(getattr(h, "baseFilename", "")) == Path(log_file)
        for h in logger.handlers
    )
    if not file_handler_exists:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger
