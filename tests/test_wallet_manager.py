from pathlib import Path
import yaml
import pytest

from crypto_bot import wallet_manager


def test_environment_overrides_config(tmp_path, monkeypatch):
    cfg = tmp_path / "user_config.yaml"
    cfg.write_text(
        "exchange: coinbase\ncoinbase_api_key: file_key\ncoinbase_api_secret: file_secret\n"
    )
    def fake_load(stream):
        text = stream.read() if hasattr(stream, "read") else str(stream)
        return dict(line.split(": ", 1) for line in text.strip().splitlines())
    monkeypatch.setattr(wallet_manager.yaml, "safe_load", fake_load)
    monkeypatch.setattr(wallet_manager, "CONFIG_FILE", cfg)
    monkeypatch.setenv("COINBASE_API_KEY", "env_key")
    creds = wallet_manager.load_or_create()
    assert creds["coinbase_api_key"] == "env_key"


def test_loads_from_legacy_path(tmp_path, monkeypatch):
    legacy_dir = tmp_path / ".cointrader"
    legacy_dir.mkdir()
    legacy_cfg = legacy_dir / "user_config.yaml"
    legacy_cfg.write_text(
        "exchange: coinbase\ncoinbase_api_key: legacy_key\ncoinbase_api_secret: legacy_secret\n"
    )
    def fake_load(stream):
        text = stream.read() if hasattr(stream, "read") else str(stream)
        return dict(line.split(": ", 1) for line in text.strip().splitlines())
    monkeypatch.setattr(wallet_manager.yaml, "safe_load", fake_load)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(wallet_manager, "CONFIG_FILE", tmp_path / "user_config.yaml")
    monkeypatch.setattr(wallet_manager, "LEGACY_CONFIG_FILE", legacy_cfg)
    monkeypatch.delenv("COINBASE_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    creds = wallet_manager.load_or_create()
    assert creds["coinbase_api_key"] == "legacy_key"


def test_sanitize_secret_adds_padding():
    secret = "YWJjZA"
    padded = wallet_manager._sanitize_secret(secret)
    assert padded == "YWJjZA=="


def test_get_wallet_missing(monkeypatch):
    monkeypatch.delenv("SOLANA_PRIVATE_KEY", raising=False)
    with pytest.raises(ValueError):
        wallet_manager.get_wallet()


def test_get_wallet_invalid(monkeypatch):
    monkeypatch.setenv("SOLANA_PRIVATE_KEY", "not-json")
    with pytest.raises(ValueError):
        wallet_manager.get_wallet()


def test_get_wallet_success(monkeypatch):
    monkeypatch.setenv("SOLANA_PRIVATE_KEY", "[1,2,3,4]")

    class KP:
        called = None

        @staticmethod
        def from_secret_key(b):
            KP.called = b
            return "k"

    import sys, types
    sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
    monkeypatch.setattr(sys.modules["solana.keypair"], "Keypair", KP, raising=False)
    key = wallet_manager.get_wallet()
    assert key == "k"
    assert KP.called == bytes([1, 2, 3, 4])

