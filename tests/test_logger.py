
import logging

from crypto_bot.utils.logger import setup_logger


def test_setup_logger_creates_file_and_logs_to_console(tmp_path, caplog):
    log_file = tmp_path / "test.log"
    with caplog.at_level(logging.INFO):
        logger = setup_logger("test_logger", str(log_file))
        logger.info("hello")
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.flush()

    assert log_file.exists()
    assert "hello" in log_file.read_text()
    assert any("hello" in r.getMessage() for r in caplog.records)
