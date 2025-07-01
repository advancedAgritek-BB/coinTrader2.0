import logging
from pathlib import Path


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Return a logger configured to write to ``log_file``.

    If the logger already has a ``FileHandler`` for the same file, no new
    handler is added.
    """

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(getattr(handler, "baseFilename", "")) == Path(log_file):
            return logger

    fh = logging.FileHandler(log_file)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
