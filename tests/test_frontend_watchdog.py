import pytest
import subprocess

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


class DummyProc:
    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self):
        pass


def test_start_stop_bot_endpoints(monkeypatch):
    client = app.app.test_client()
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: DummyProc())
    app.bot_proc = None
    resp = client.post('/start_bot')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'started'
    assert data['running'] is True
    assert 'uptime' in data

    app.bot_proc = DummyProc()
    resp = client.post('/stop_bot')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'stopped'
    assert data['running'] is False
    assert 'uptime' in data
