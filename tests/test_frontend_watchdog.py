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


def test_scans_route(tmp_path, monkeypatch):
    score_file = tmp_path / "scores.json"
    score_file.write_text('{"BTC": 0.5}')
    monkeypatch.setattr(app, "SCAN_FILE", score_file)
    client = app.app.test_client()
    resp = client.get("/scans")
    assert resp.status_code == 200
    assert b"BTC" in resp.data
