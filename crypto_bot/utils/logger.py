import logging
from pathlib import Path

# Default directory for all log files used across the project
LOG_DIR = Path(__file__).resolve().parents[2] / "crypto_bot" / "logs"


def setup_logger(
    name: str, log_file: str | Path | None = None, to_console: bool = True
) -> logging.Logger:
    """Return a logger configured to write to ``log_file`` within ``LOG_DIR``.

    The directory ``LOG_DIR`` is created automatically when the logger is initialized.
    If ``log_file`` is ``None``, a file named ``{name}.log`` within ``LOG_DIR`` is used.
    """

    if log_file is None:
        log_file = LOG_DIR / f"{name}.log"
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

    if to_console and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger
