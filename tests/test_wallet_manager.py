import os
import sys
import types
import yaml
import pytest


# Stub out commit_lock to avoid syntax errors during import
sys.modules.setdefault(
    "crypto_bot.utils.commit_lock", types.SimpleNamespace(check_and_update=lambda *a, **k: None)
)

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
    monkeypatch.delenv("COINBASE_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("KRAKEN_API_KEY", raising=False)
    creds = wallet_manager.load_or_create()
    assert creds["coinbase_api_key"] == "cb_key"
    assert creds["kraken_api_key"] == "kr_key"


def test_load_exports_helius_api_key(tmp_path, monkeypatch):
    cfg = tmp_path / "user_config.yaml"
    data = {"helius_api_key": "hk"}
    cfg.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(wallet_manager, "CONFIG_FILE", cfg)
    monkeypatch.delenv("HELIUS_API_KEY", raising=False)
    monkeypatch.delenv("HELIUS_KEY", raising=False)
    wallet_manager.load_or_create()
    assert os.environ["HELIUS_API_KEY"] == "hk"
    assert os.environ["HELIUS_KEY"] == "hk"


def test_load_exports_lunarcrush_key(tmp_path, monkeypatch):
    cfg = tmp_path / "user_config.yaml"
    data = {"lunarcrush_api_key": "lk"}
    cfg.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(wallet_manager, "CONFIG_FILE", cfg)
    monkeypatch.delenv("LUNARCRUSH_API_KEY", raising=False)
    creds = wallet_manager.load_or_create()
    assert os.environ["LUNARCRUSH_API_KEY"] == "lk"
    assert creds["lunarcrush_api_key"] == "lk"


def test_load_exports_supabase_creds(tmp_path, monkeypatch):
    cfg = tmp_path / "user_config.yaml"
    data = {"supabase_url": "url", "supabase_key": "key"}
    cfg.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(wallet_manager, "CONFIG_FILE", cfg)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    creds = wallet_manager.load_or_create()
    assert os.environ["SUPABASE_URL"] == "url"
    assert os.environ["SUPABASE_KEY"] == "key"
    assert os.environ["SUPABASE_SERVICE_ROLE_KEY"] == "key"
    assert creds["supabase_url"] == "url"
    assert creds["supabase_key"] == "key"


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

