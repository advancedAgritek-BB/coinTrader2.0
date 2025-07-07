import time
import telegram_ctl as ctl


def test_set_get_page():
    ctl.callback_state.clear()
    ctl.set_page(1, "logs", 2)
    assert ctl.get_page(1, "logs") == 2
    assert "1" in ctl.callback_state


def test_expiration(monkeypatch):
    ctl.callback_state.clear()
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    ctl.set_page(1, "logs", 1)
    assert ctl.get_page(1, "logs") == 1

    monkeypatch.setattr(time, "time", lambda: now + ctl.callback_timeout + 1)
    assert ctl.get_page(1, "logs") == 0
    assert ctl.callback_state == {}
