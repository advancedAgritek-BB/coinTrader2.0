import logging
import logging.config
from pathlib import Path


def setup_logging(log_dir='crypto_bot/logs'):
    """Configure logging for the application using RotatingFileHandler."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    config = {
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'rotating_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'standard',
                'filename': str(log_path / 'bot.log'),
                'maxBytes': 1024 * 1024,
                'backupCount': 3,
            },
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
        },
        'root': {
            'handlers': ['rotating_file', 'console'],
            'level': 'INFO',
        },
    }
    logging.config.dictConfig(config)
