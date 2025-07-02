import yaml

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

