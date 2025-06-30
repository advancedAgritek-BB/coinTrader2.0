import logging
from pathlib import Path


def setup_logger(name: str, log_file: str) -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
