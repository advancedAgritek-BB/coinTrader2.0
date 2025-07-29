import sys
import types
import yaml
import pytest

from crypto_bot import wallet_manager


@pytest.fixture(autouse=True)
def stub_solana_keypair(monkeypatch):
    """Ensure ``solana.keypair`` is available even if Solana isn't installed."""
    module = sys.modules.setdefault("solana.keypair", types.ModuleType("solana.keypair"))
    if not hasattr(module, "Keypair"):
        class Dummy:
            @staticmethod
            def from_secret_key(secret):
                return None

        monkeypatch.setattr(module, "Keypair", Dummy, raising=False)
    for var in (
        "API_KEY",
        "API_SECRET",
        "API_PASSPHRASE",
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET",
        "COINBASE_API_PASSPHRASE",
        "KRAKEN_API_KEY",
        "KRAKEN_API_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


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


def test_load_returns_both_exchange_creds(tmp_path, monkeypatch):
    cfg = tmp_path / "user_config.yaml"
    data = {
        "coinbase_api_key": "cb_key",
        "coinbase_api_secret": "cb_secret",
        "coinbase_passphrase": "pass",
        "kraken_api_key": "kr_key",
        "kraken_api_secret": "kr_secret",
    }
    cfg.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(wallet_manager, "CONFIG_FILE", cfg)
    creds = wallet_manager.load_or_create()
    assert creds["coinbase_api_key"] == "cb_key"
    assert creds["kraken_api_key"] == "kr_key"


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


def test_env_creds_no_config(tmp_path, monkeypatch):
    cfg = tmp_path / "missing.yaml"
    monkeypatch.setattr(wallet_manager, "CONFIG_FILE", cfg)
    for k, v in {
        "COINBASE_API_KEY": "env_key",
        "COINBASE_API_SECRET": "ZW52X3NlY3JldA==",
        "COINBASE_API_PASSPHRASE": "env_pass",
        "KRAKEN_API_KEY": "kr_key",
        "KRAKEN_API_SECRET": "a3Jfc2VjcmV0",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("API_KEY", "env_key")
    monkeypatch.setenv("API_SECRET", "env_secret")

    def fail(*args, **kwargs):
        raise AssertionError("input called")

    monkeypatch.setattr("builtins.input", fail)
    creds = wallet_manager.load_or_create()
    assert creds["coinbase_api_key"] == "env_key"
    assert creds["coinbase_api_secret"] == "ZW52X3NlY3JldA=="
    assert creds["coinbase_passphrase"] == "env_pass"
    assert creds["kraken_api_key"] == "kr_key"
    assert creds["kraken_api_secret"] == "a3Jfc2VjcmV0"
    assert not cfg.exists()


def test_encryption_roundtrip(tmp_path, monkeypatch):
    cfg = tmp_path / "user_config.yaml"
    monkeypatch.setattr(wallet_manager, "CONFIG_FILE", cfg)

    from cryptography.fernet import Fernet

    key = Fernet.generate_key()
    f = Fernet(key)
    monkeypatch.setattr(wallet_manager, "_fernet", f, raising=False)
    monkeypatch.setattr(wallet_manager, "FERNET_KEY", key.decode(), raising=False)

    monkeypatch.delenv("COINBASE_API_KEY", raising=False)
    monkeypatch.delenv("COINBASE_API_SECRET", raising=False)
    monkeypatch.delenv("COINBASE_API_PASSPHRASE", raising=False)
    monkeypatch.delenv("KRAKEN_API_KEY", raising=False)
    monkeypatch.delenv("KRAKEN_API_SECRET", raising=False)

    monkeypatch.setattr(wallet_manager, "prompt_user", lambda: {"coinbase_api_key": "abc"})

    creds = wallet_manager.load_or_create()
    assert creds["coinbase_api_key"] == "abc"
    text = cfg.read_text()
    loaded = yaml.safe_load(text)
    assert loaded["coinbase_api_key"] != "abc"
    assert wallet_manager._decrypt(loaded["coinbase_api_key"]) == "abc"

    # Loading again should decrypt automatically
    creds2 = wallet_manager.load_or_create()
    assert creds2["coinbase_api_key"] == "abc"


def test_invalid_fernet_key(tmp_path, monkeypatch):
    cfg = tmp_path / "user_config.yaml"
    data = {"coinbase_api_key": "abc"}
    cfg.write_text(yaml.safe_dump(data))

    monkeypatch.setenv("FERNET_KEY", "invalid")
    import importlib
    import crypto_bot.wallet_manager as wm
    wm = importlib.reload(wm)
    monkeypatch.setattr(wm, "CONFIG_FILE", cfg)

    creds = wm.load_or_create()
    assert creds["coinbase_api_key"] == "abc"
    assert yaml.safe_load(cfg.read_text())["coinbase_api_key"] == "abc"

    monkeypatch.delenv("FERNET_KEY", raising=False)
    importlib.reload(wm)

