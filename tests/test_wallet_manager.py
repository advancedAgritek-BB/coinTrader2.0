import yaml
import pytest

from crypto_bot import wallet_manager


def test_environment_overrides_config(tmp_path, monkeypatch):
    cfg = tmp_path / "user_config.yaml"
    data = {
        "exchange": "coinbase",
        "coinbase_api_key": "file_key",
        "coinbase_api_secret": "file_secret",
    }
    cfg.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(wallet_manager, "CONFIG_FILE", cfg)
    monkeypatch.setenv("COINBASE_API_KEY", "env_key")
    creds = wallet_manager.load_or_create()
    assert creds["coinbase_api_key"] == "env_key"


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

