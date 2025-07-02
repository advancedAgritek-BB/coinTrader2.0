
import logging
import warnings

from crypto_bot.utils.telegram import send_message

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


def test_send_message_async_no_warning(monkeypatch):
    calls = {}

    class DummyBot:
        def __init__(self, token):
            calls["token"] = token

        async def send_message(self, chat_id, text):
            calls["chat_id"] = chat_id
            calls["text"] = text

    monkeypatch.setattr("crypto_bot.utils.telegram.Bot", DummyBot)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")
        err = send_message("t", "c", "msg")

    assert err is None
    assert calls["chat_id"] == "c"
    assert calls["text"] == "msg"
    assert w == []
