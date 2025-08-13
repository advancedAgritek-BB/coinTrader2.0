import os
import subprocess
import sys

import pytest

import crypto_bot.main as main


def test_wizard_launch(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def fake_run(cmd, *a, **k):
        calls.append(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    cfg = tmp_path / "user_config.yaml"
    monkeypatch.setattr(main, "USER_CONFIG_PATH", cfg)
    for var in main.REQUIRED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    def fake_call(cmd):
        calls.append(cmd)
        cfg.write_text("api_key: x")
        for var in main.REQUIRED_ENV_VARS:
            monkeypatch.setenv(var, "x")
        monkeypatch.setenv("API_KEY", "x")
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)

    main._ensure_user_setup()

    expected = [sys.executable, "-m", "crypto_bot.wallet_manager"]
    assert calls and calls[0] == expected
    assert calls and calls[0] == [sys.executable, "-m", "crypto_bot.wallet_manager"]


def test_no_launch_when_configured(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    def fake_call(cmd):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    cfg = tmp_path / "user_config.yaml"
    cfg.write_text("dummy: 1")
    monkeypatch.setattr(main, "USER_CONFIG_PATH", cfg)
    for var in main.REQUIRED_ENV_VARS:
        monkeypatch.setenv(var, "x")

    main._ensure_user_setup()

    assert not calls


def test_headless_exit(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    exit_code: dict[str, int] = {}

    def fake_exit(code=0):
        exit_code["code"] = code
        raise SystemExit

    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit):
        main._run_wallet_manager()

    assert exit_code.get("code") == 2
    assert "Interactive setup required" in capsys.readouterr().err

