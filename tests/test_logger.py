
import logging
import warnings
import asyncio
import pytest

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

    monkeypatch.setattr("crypto_bot.utils.telegram.telegram.Bot", DummyBot)

    async def runner():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")
            await send_message("t", "c", "msg")
        return w

    w = asyncio.run(runner())

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
        with pytest.raises(RuntimeError):
            asyncio.run(telegram.send_message("t", "c", "msg"))

    assert any("boom" in r.getMessage() for r in caplog.records)


def test_send_message_exception_raises(monkeypatch, tmp_path, caplog):
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
        with pytest.raises(RuntimeError):
            await telegram.send_message("t", "c", "msg")

    with caplog.at_level(logging.ERROR):
        asyncio.run(runner())

    assert any("boom" in r.getMessage() for r in caplog.records)


def test_send_test_message_success(monkeypatch):
    calls = {}

    async def fake_send(token, chat_id, text):
        calls["args"] = (token, chat_id, text)

    import crypto_bot.utils.telegram as telegram

    monkeypatch.setattr(telegram, "send_message", fake_send)

    assert asyncio.run(telegram.send_test_message("t", "c", "hello")) is True
    monkeypatch.setattr(telegram, "send_message_sync", fake_send)
    assert telegram.send_test_message("t", "c", "hello") is True
    assert calls["args"] == ("t", "c", "hello")


def test_send_test_message_invalid_token(monkeypatch):
    import crypto_bot.utils.telegram as telegram

    async def fake_send(token, chat_id, text):
        raise ValueError("bad token")

    monkeypatch.setattr(telegram, "send_message", fake_send)

    with pytest.raises(ValueError):
        asyncio.run(telegram.send_test_message("t", "c"))


def test_send_test_message_timeout_retry(monkeypatch):
    import crypto_bot.utils.telegram as telegram

    calls = {"count": 0}

    async def fake_send(token, chat_id, text):
        calls["count"] += 1
        if calls["count"] == 1:
            raise asyncio.TimeoutError

    monkeypatch.setattr(telegram, "send_message", fake_send)
    def boom(*a, **k):
        raise RuntimeError("err")

    monkeypatch.setattr(telegram, "send_message_sync", boom)

    assert (
        asyncio.run(
            telegram.send_test_message("t", "c", retries=2, retry_delay=0)
        )
        is True
    )
    assert calls["count"] == 2


def test_notifier_stops_after_failure(monkeypatch, caplog):
    import crypto_bot.utils.telegram as telegram

    calls = {"count": 0}

    async def fake_send(token, chat_id, text):
        calls["count"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(telegram, "send_message_sync", fake_send)
    monkeypatch.setattr(telegram.asyncio, "get_running_loop", lambda: object())
    monkeypatch.setattr(telegram, "logger", logging.getLogger("test_notifier"))

    notifier = telegram.TelegramNotifier(True, "t", "c")

    with caplog.at_level(logging.ERROR):
        err1 = notifier.notify("msg1")
        err2 = notifier.notify("msg2")

    assert err1 == "boom"
    assert err2 is None
    assert calls["count"] == 1
    assert any("disabling telegram notifications" in r.getMessage().lower() for r in caplog.records)


def test_notifier_rate_limits(monkeypatch):
    import types
    import crypto_bot.utils.telegram as telegram

    current = {"t": 0.0}

    def fake_time():
        return current["t"]

    def fake_sleep(seconds):
        current["t"] += seconds

    send_times = []

    async def fake_send(token, chat_id, text):
        send_times.append(current["t"])

    monkeypatch.setattr(telegram, "send_message_sync", fake_send)
    monkeypatch.setattr(telegram, "time", types.SimpleNamespace(time=fake_time, sleep=fake_sleep))

    notifier = telegram.TelegramNotifier(True, "t", "c")

    for _ in range(25):
        notifier.notify("msg")

    # ensure at least one second between sends
    assert all(b - a >= 1.0 for a, b in zip(send_times, send_times[1:]))

    # ensure no more than 20 messages in any 60 second window
    for i in range(len(send_times) - 20):
        assert send_times[i + 20] - send_times[i] >= 60
