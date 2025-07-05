
import logging
import warnings
import asyncio

from crypto_bot.utils.telegram import send_message

from crypto_bot.utils.logger import setup_logger


def test_setup_logger_creates_file_and_logs_to_console_by_default(tmp_path, caplog):
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


def test_setup_logger_can_disable_console_output(tmp_path, caplog):
    log_file = tmp_path / "test_no_console.log"
    with caplog.at_level(logging.INFO):
        logger = setup_logger("test_no_console", str(log_file), to_console=False)
        logger.info("hello")
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.flush()

    assert log_file.exists()
    assert "hello" in log_file.read_text()
    assert not any(type(h) is logging.StreamHandler for h in logger.handlers)


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


def test_send_message_exception_logged(monkeypatch, tmp_path, caplog):
    class DummyBot:
        def __init__(self, token):
            pass

        def send_message(self, chat_id, text):
            raise RuntimeError("boom")

    import crypto_bot.utils.telegram as telegram

    monkeypatch.setattr(telegram, "Bot", DummyBot)

    log_file = tmp_path / "bot.log"
    logger = setup_logger("tel_test", str(log_file))
    monkeypatch.setattr(telegram, "logger", logger)

    with caplog.at_level(logging.ERROR):
        err = telegram.send_message("t", "c", "msg")

    assert err == "boom"
    assert any("boom" in r.getMessage() for r in caplog.records)


def test_send_message_async_exception_logged(monkeypatch, tmp_path, caplog):
    class DummyBot:
        def __init__(self, token):
            pass

        async def send_message(self, chat_id, text):
            raise RuntimeError("boom")

    import crypto_bot.utils.telegram as telegram

    monkeypatch.setattr(telegram, "Bot", DummyBot)

    log_file = tmp_path / "bot.log"
    logger = setup_logger("tel_test_async", str(log_file))
    monkeypatch.setattr(telegram, "logger", logger)

    async def runner():
        err = telegram.send_message("t", "c", "msg")
        await asyncio.sleep(0)
        return err

    with caplog.at_level(logging.ERROR):
        err = asyncio.run(runner())

    assert err is None
    assert any("boom" in r.getMessage() for r in caplog.records)


def test_send_test_message_success(monkeypatch):
    calls = {}

    def fake_send(token, chat_id, text):
        calls["args"] = (token, chat_id, text)
        return None

    import crypto_bot.utils.telegram as telegram

    monkeypatch.setattr(telegram, "send_message", fake_send)

    assert telegram.send_test_message("t", "c", "hello") is True
    assert calls["args"] == ("t", "c", "hello")


def test_send_test_message_failure(monkeypatch):
    import crypto_bot.utils.telegram as telegram

    monkeypatch.setattr(telegram, "send_message", lambda *a, **k: "err")

    assert telegram.send_test_message("t", "c") is False


def test_notifier_stops_after_failure(monkeypatch, caplog):
    import crypto_bot.utils.telegram as telegram

    calls = {"count": 0}

    def fake_send(token, chat_id, text):
        calls["count"] += 1
        return "boom"

    monkeypatch.setattr(telegram, "send_message", fake_send)
    monkeypatch.setattr(telegram, "logger", logging.getLogger("test_notifier"))

    notifier = telegram.TelegramNotifier(True, "t", "c")

    with caplog.at_level(logging.ERROR):
        err1 = notifier.notify("msg1")
        err2 = notifier.notify("msg2")

    assert err1 == "boom"
    assert err2 is None
    assert calls["count"] == 1
    assert any("disabling telegram notifications" in r.getMessage().lower() for r in caplog.records)
