import yaml
import importlib.util
from pathlib import Path

if not hasattr(yaml, "__file__"):
    import sys
    sys.modules.pop("yaml", None)
    spec = importlib.util.find_spec("yaml")
    if spec and spec.loader:
        real_yaml = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real_yaml)
        yaml = real_yaml

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
    assert "indicator_lookback" in config
    assert "rsi_overbought_pct" in config
    assert "rsi_oversold_pct" in config
    assert "bb_squeeze_pct" in config
    assert "adx_threshold" in config
    assert "sl_mult" in config
    assert "tp_mult" in config
    assert "allow_short" in config
    assert "ml_enabled" in config
    assert "scan_lookback_limit" in config
    assert "cycle_lookback_limit" in config
    assert "top_n_symbols" in config
    assert "min_confidence_score" in config
    assert "signal_fusion" in config
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
        "dynamic_grid",
        "use_ml_center",
        "min_range_pct",
        "leverage",
        "arbitrage_pairs",
        "arbitrage_threshold",
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

    assert "telegram" in config
    assert "command_cooldown" in config["telegram"]
    assert "solana_scanner" in config
    sol_scanner = config["solana_scanner"]
    assert isinstance(sol_scanner, dict)
    for key in [
        "enabled",
        "interval_minutes",
        "api_keys",
        "min_volume_usd",
        "max_tokens_per_scan",
    ]:
        assert key in sol_scanner

    assert "pyth" in config
    pyth_cfg = config["pyth"]
    assert isinstance(pyth_cfg, dict)
    for key in [
        "enabled",
        "solana_endpoint",
        "solana_ws_endpoint",
        "program_id",
    ]:
        assert key in pyth_cfg


def test_load_config_normalizes_symbol(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("scan_markets: true\nsymbol: XBT/USDT\n")
    import types, crypto_bot.main as main

    def _simple_yaml(f):
        data = {}
        for line in f.read().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
        return data

    monkeypatch.setattr(main, "CONFIG_PATH", path)
    monkeypatch.setattr(main, "yaml", types.SimpleNamespace(safe_load=_simple_yaml))
    loaded = main.load_config()
    assert loaded["symbol"] == "BTC/USDT"
