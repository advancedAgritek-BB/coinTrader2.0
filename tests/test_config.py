import yaml
from pathlib import Path

CONFIG_PATH = Path("crypto_bot/config.yaml")


def test_load_config_returns_dict():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict)
    assert "mode" in config
    assert "testing_mode" in config
    assert "risk" in config
    assert "min_cooldown" in config
    assert "atr_normalization" in config
    assert "top_n_symbols" in config
    assert "min_confidence_score" in config
    assert "voting_strategies" in config
    assert "min_agreeing_votes" in config
    assert "ohlcv_timeout" in config
    assert "max_ohlcv_failures" in config
