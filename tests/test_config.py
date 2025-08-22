import yaml
import importlib.util
from pathlib import Path
import asyncio
import types
import os

from crypto_bot import main
from crypto_bot.regime import regime_classifier as rc

if not hasattr(yaml, "__file__"):
    import sys

    sys.modules.pop("yaml", None)
    spec = importlib.util.find_spec("yaml")
    if spec and spec.loader:
        real_yaml = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real_yaml)
        yaml = real_yaml
    sys.modules.setdefault("yaml", yaml)

CONFIG_PATH = Path("crypto_bot/config.yaml")


def _import_main(monkeypatch):
    import sys

    sys.modules.setdefault("yaml", yaml)
    sys.modules.setdefault("redis", types.SimpleNamespace())
    sys.modules.setdefault(
        "crypto_bot.solana",
        types.SimpleNamespace(get_solana_new_tokens=lambda *_a, **_k: []),
    )
    sys.modules.setdefault("crypto_bot.solana.scalping", types.SimpleNamespace())
    sys.modules.setdefault(
        "crypto_bot.solana.exit",
        types.SimpleNamespace(monitor_price=lambda *_a, **_k: None),
    )
    sys.modules.setdefault(
        "crypto_bot.execution.solana_mempool",
        types.SimpleNamespace(SolanaMempoolMonitor=object),
    )
    sys.modules.setdefault(
        "crypto_bot.regime.pattern_detector",
        types.SimpleNamespace(detect_patterns=lambda *_a, **_k: {}),
    )
    sys.modules.setdefault(
        "crypto_bot.utils.market_analyzer",
        types.SimpleNamespace(analyze_symbol=lambda *_a, **_k: None),
    )
    sys.modules.setdefault(
        "crypto_bot.strategy_router",
        types.SimpleNamespace(strategy_for=lambda *_a, **_k: None),
    )
    sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))
    sys.modules.setdefault("gspread", types.SimpleNamespace(authorize=lambda *a, **k: None))
    sys.modules.setdefault(
        "oauth2client.service_account",
        types.SimpleNamespace(
            ServiceAccountCredentials=types.SimpleNamespace(
                from_json_keyfile_name=lambda *a, **k: None
            )
        ),
    )
    sys.modules.setdefault("rich.console", types.SimpleNamespace(Console=object))
    sys.modules.setdefault("rich.table", types.SimpleNamespace(Table=object))
    sys.modules.setdefault(
        "crypto_bot.utils.symbol_pre_filter",
        types.SimpleNamespace(filter_symbols=lambda *_a, **_k: ([], [])),
    )
    class _FakeGen:
        pass

    sys.modules.setdefault(
        "numpy.random",
        types.SimpleNamespace(default_rng=lambda *_a, **_k: _FakeGen(), Generator=_FakeGen),
    )

    import crypto_bot.main as main

    return main


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
    assert "arbitrage_threshold" in config
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
    voting = config["voting_strategies"]
    assert isinstance(voting, dict)
    assert "strategies" in voting
    assert "min_agreeing_votes" in voting
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

    assert "flash_crash_bot" in config
    fc_bot = config["flash_crash_bot"]
    assert isinstance(fc_bot, dict)
    for key in ["drop_thr", "vol_thr", "tp_thr"]:
        assert key in fc_bot
    assert "cross_chain_arb_bot" in config
    cca = config["cross_chain_arb_bot"]
    assert isinstance(cca, dict)
    for key in ["pair", "spread_threshold", "fee_threshold"]:
        assert key in cca

    assert "dca" in config
    dca_cfg = config["dca"]
    assert isinstance(dca_cfg, dict)
    for key in ["enabled", "max_entries", "size_multiplier"]:
        assert key in dca_cfg

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
    assert "pyth_quotes" in config
    assert config["pyth_quotes"]

    assert "auto_convert_quote" in config


def test_exit_config_unified():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    assert "exit_strategy" in config
    exit_cfg = config["exit_strategy"]
    assert isinstance(exit_cfg, dict)
    for key in [
        "fib_tp_enabled",
        "min_gain_to_trail",
        "partial_levels",
        "scale_out",
        "trailing_stop_factor",
        "trailing_stop_pct",
        "default_sl_pct",
        "default_tp_pct",
    ]:
        assert key in exit_cfg
    assert "exits" not in config


def test_load_config_normalizes_symbol(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("scan_markets: true\nsymbol: XBT/USDT\n")
    import types
    main = _import_main(monkeypatch)
    path.write_text("scan_markets: true\nsymbol: XBT/USDT\nml_enabled: false\n")
    import types, sys

    # Ensure stub modules are available before importing main
    sys.modules.setdefault("yaml", yaml)
    sys.modules.setdefault("redis", types.SimpleNamespace())
    sys.modules.setdefault(
        "crypto_bot.solana",
        types.SimpleNamespace(get_solana_new_tokens=lambda *_a, **_k: []),
    )
    sys.modules.setdefault("crypto_bot.solana.scalping", types.SimpleNamespace())
    sys.modules.setdefault(
        "crypto_bot.solana.exit",
        types.SimpleNamespace(monitor_price=lambda *_a, **_k: None),
    )
    sys.modules.setdefault(
        "crypto_bot.execution.solana_mempool",
        types.SimpleNamespace(SolanaMempoolMonitor=object),
    )
    sys.modules.setdefault(
        "crypto_bot.auto_optimizer", types.SimpleNamespace(optimize_strategies=lambda *_a, **_k: None)
    )
    sys.modules.setdefault(
        "crypto_bot.regime.regime_classifier",
        types.SimpleNamespace(
            classify_regime_async=lambda *_a, **_k: None,
            classify_regime_cached=lambda *_a, **_k: None,
        ),
    )
    sys.modules.setdefault(
        "crypto_bot.utils.market_analyzer",
        types.SimpleNamespace(analyze_symbol=lambda *_a, **_k: None),
    )
    sys.modules.setdefault(
        "crypto_bot.strategy_router",
        types.SimpleNamespace(strategy_for=lambda *_a, **_k: None),
    )
    sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))
    sys.modules.setdefault("gspread", types.SimpleNamespace(authorize=lambda *a, **k: None))
    sys.modules.setdefault(
        "oauth2client.service_account",
        types.SimpleNamespace(
            ServiceAccountCredentials=types.SimpleNamespace(
                from_json_keyfile_name=lambda *a, **k: None
            )
        ),
    )
    sys.modules.setdefault("rich.console", types.SimpleNamespace(Console=object))
    sys.modules.setdefault("rich.table", types.SimpleNamespace(Table=object))
    sys.modules.setdefault(
        "crypto_bot.utils.symbol_pre_filter",
        types.SimpleNamespace(filter_symbols=lambda *_a, **_k: ([], [])),
    )
    class _FakeGen:
        pass
    sys.modules.setdefault(
        "numpy.random",
        types.SimpleNamespace(default_rng=lambda *_a, **_k: _FakeGen(), Generator=_FakeGen),
    )

    import crypto_bot.main as main

    def _simple_yaml(f):
        data = {}
        for line in f.read().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                val = v.strip()
                if val.lower() in {"true", "false"}:
                    data[k.strip()] = val.lower() == "true"
                else:
                    data[k.strip()] = val
        return data

    monkeypatch.setattr(main, "CONFIG_PATH", path)
    monkeypatch.setattr(main, "yaml", types.SimpleNamespace(safe_load=_simple_yaml))
    loaded = main.load_config()
    assert loaded["symbol"] == "BTCUSDT"


def test_reload_config_clears_symbol_cache(monkeypatch, tmp_path):
    import json

    main = _import_main(monkeypatch)
    from crypto_bot.utils import symbol_utils

    # Pre-populate symbol cache and hash
    symbol_utils._cached_symbols = ([("ETH/USD", 1.0)], [])
    symbol_utils._last_refresh = 123.0
    symbol_utils._cached_hash = "oldhash"
    cache_file = tmp_path / "symcache.json"
    cache_file.write_text(
        json.dumps({"timestamp": 0, "hash": "oldhash", "symbols": [], "onchain": []})
    )
    monkeypatch.setattr(symbol_utils, "SYMBOL_CACHE_FILE", cache_file)

    def fake_load_config():
        return {
            "symbol": "BTCUSDT",
            "risk": {
                "max_drawdown": 1.0,
                "stop_loss_pct": 0.0,
                "take_profit_pct": 0.0,
            },
        }

    monkeypatch.setattr(main, "load_config", fake_load_config)

    config = {}
    ctx = main.BotContext({}, {}, {}, config)
    risk_manager = main.RiskManager(main.RiskConfig(1.0, 0.0, 0.0))
    rotator = types.SimpleNamespace(config={})
    guard = main.OpenPositionGuard(1)

    asyncio.run(
        main.reload_config(config, ctx, risk_manager, rotator, guard, force=True)
    )

    assert symbol_utils._cached_symbols is None
    assert symbol_utils._last_refresh == 0.0
    assert not cache_file.exists()


def test_load_config_async_detects_section_changes(tmp_path, monkeypatch):
    main = _import_main(monkeypatch)
    monkeypatch.setattr(main.ml_utils, "init_ml_components", lambda: (True, ""))
    monkeypatch.setattr(main.ml_utils, "ML_AVAILABLE", True, raising=False)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("ml_enabled: true\nrisk:\n  max_drawdown: 1\n")
    monkeypatch.setattr(main, "CONFIG_PATH", cfg_path)
    main._CONFIG_CACHE.clear()
    main._CONFIG_MTIMES.clear()
    main._LAST_ML_CFG = None

    dummy = types.SimpleNamespace(dict=lambda: {})
    monkeypatch.setattr(main, "ScannerConfig", types.SimpleNamespace(model_validate=lambda d: None))
    monkeypatch.setattr(main, "SolanaScannerConfig", types.SimpleNamespace(model_validate=lambda d: dummy))
    monkeypatch.setattr(main, "PythConfig", types.SimpleNamespace(model_validate=lambda d: dummy))

    cfg, changed = asyncio.run(main.load_config_async())
    assert "risk" in changed and "ml_enabled" in changed

    _, changed2 = asyncio.run(main.load_config_async())
    assert changed2 == set()

    cfg_path.write_text("ml_enabled: true\nrisk:\n  max_drawdown: 2\n")
    os.utime(cfg_path, None)
    _, changed3 = asyncio.run(main.load_config_async())
    assert "risk" in changed3


def test_ensure_ml_only_on_change(tmp_path, monkeypatch):
    main = _import_main(monkeypatch)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("ml_enabled: true\n")
    monkeypatch.setattr(main, "CONFIG_PATH", cfg_path)
    main._CONFIG_CACHE.clear()
    main._CONFIG_MTIMES.clear()
    main._LAST_ML_CFG = None

    dummy = types.SimpleNamespace(dict=lambda: {})
    monkeypatch.setattr(main, "ScannerConfig", types.SimpleNamespace(model_validate=lambda d: None))
    monkeypatch.setattr(main, "SolanaScannerConfig", types.SimpleNamespace(model_validate=lambda d: dummy))
    monkeypatch.setattr(main, "PythConfig", types.SimpleNamespace(model_validate=lambda d: dummy))

    calls: list[bool] = []

    def fake_load(symbol):
        calls.append(True)
        return object(), None, "path"

    monkeypatch.setattr(main, "load_regime_model", fake_load)
    monkeypatch.setattr(main.ml_utils, "init_ml_components", lambda: (True, ""))
    monkeypatch.setattr(main.ml_utils, "ML_AVAILABLE", True)

    asyncio.run(main.load_config_async())
    assert calls == [True]

    calls.clear()
    asyncio.run(main.load_config_async())
    assert calls == []

    cfg_path.write_text("ml_enabled: true\n")
    os.utime(cfg_path, None)
    calls.clear()
    asyncio.run(main.load_config_async())
    assert calls == []
