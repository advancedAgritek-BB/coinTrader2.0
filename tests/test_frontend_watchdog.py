import pytest

from frontend import app


class StopLoop(Exception):
    pass


def test_watchdog_thread_start(monkeypatch):
    """watch_bot should access bot_proc without NameError."""
    def stop(_):
        raise StopLoop

    monkeypatch.setattr(app.time, "sleep", stop)
    app.bot_proc = None
    with pytest.raises(StopLoop):
        app.watch_bot()
