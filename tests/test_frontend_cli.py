from frontend import app
import subprocess


def test_cli_get():
    client = app.app.test_client()
    resp = client.get('/cli')
    assert resp.status_code == 200


def test_cli_post_runs_command(monkeypatch):
    client = app.app.test_client()

    def fake_run(cmd, shell, capture_output, text, check):
        class Result:
            stdout = 'out'
            stderr = 'err'
        Result.returned_cmd = cmd
        return Result()

    monkeypatch.setattr(subprocess, 'run', fake_run)
    resp = client.post('/cli', data={'base': 'custom', 'command': 'echo hi'})
    assert b'out' in resp.data
    assert b'err' in resp.data


class FakeProc:
    def __init__(self):
        self.pid = 1

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self):
        pass


def test_start_stop_bot_json(monkeypatch):
    client = app.app.test_client()

    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: FakeProc())
    app.bot_proc = None
    resp = client.post('/start_bot', json={'mode': 'dry_run'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'started'
    assert data['running'] is True
    assert 'uptime' in data
    assert data['mode'] == 'dry_run'
    assert app.bot_proc is not None

    app.bot_proc = FakeProc()
    resp = client.post('/stop_bot')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'stopped'
    assert data['running'] is False
    assert 'uptime' in data
    assert app.bot_proc is None
