from crypto_bot.log_config import setup_logging
import logging

def test_setup_logging_creates_rotating_file(tmp_path):
    log_dir = tmp_path / 'logs'
    setup_logging(log_dir=log_dir)
    logger = logging.getLogger('test_logger')
    logger.info('hello')
    for h in logger.handlers:
        if hasattr(h, 'flush'):
            h.flush()
    log_file = log_dir / 'bot.log'
    assert log_file.exists()
    assert 'hello' in log_file.read_text()
