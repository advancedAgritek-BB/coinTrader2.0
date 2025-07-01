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
