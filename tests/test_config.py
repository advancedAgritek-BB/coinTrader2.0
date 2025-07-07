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
    assert "grid_bot" in config
    grid_bot = config["grid_bot"]
    assert isinstance(grid_bot, dict)
    for key in [
        "range_window",
        "atr_period",
        "spacing_factor",
        "trend_ema_fast",
        "trend_ema_slow",
        "volume_ma_window",
        "volume_multiple",
        "vol_zscore_threshold",
        "max_active_legs",
        "cooldown_bars",
        "breakout_mult",
        "atr_normalization",
    ]:
        assert key in grid_bot
