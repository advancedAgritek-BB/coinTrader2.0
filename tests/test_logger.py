from crypto_bot.utils.logger import setup_logger

def test_setup_logger_creates_file(tmp_path):
    log_file = tmp_path / 'test.log'
    logger = setup_logger('test_logger', str(log_file))
    logger.info('hello')
    logger.handlers[0].flush()
    assert log_file.exists()
    assert 'hello' in log_file.read_text()
