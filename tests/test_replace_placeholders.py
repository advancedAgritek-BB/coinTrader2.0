import sys
import types
import yaml
import asyncio


def _import_main():
    sys.modules.setdefault("yaml", yaml)
    sys.modules.setdefault("redis", types.SimpleNamespace())
    sys.modules.setdefault(
        "crypto_bot.solana",
        types.SimpleNamespace(get_solana_new_tokens=lambda *_a, **_k: []),
    )
    sys.modules.setdefault("crypto_bot.solana.scalping", types.SimpleNamespace())
    sys.modules.setdefault(
        "crypto_bot.solana.exit",
        types.SimpleNamespace(monitor_price=lambda *_a, **_k: None, quick_exit=lambda *_a, **_k: None),
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
        "crypto_bot.utils.market_analyzer", types.SimpleNamespace(analyze_symbol=lambda *_a, **_k: None)
    )
    sys.modules.setdefault(
        "crypto_bot.strategy_router", types.SimpleNamespace(strategy_for=lambda *_a, **_k: None)
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


def test_replace_placeholders(monkeypatch):
    main = _import_main()

    monkeypatch.setenv("HELIUS_KEY", "mock_key")
    cfg = {
        "solana": {
            "url": "https://api.helius.xyz/v0/?api-key=${HELIUS_KEY}",
            "nested": {"ws": "wss://mainnet.helius-rpc.com/?api-key=${HELIUS_KEY}"},
        },
        "list": ["${HELIUS_KEY}", {"deep": "value-${HELIUS_KEY}"}],
    }

    replaced = main.replace_placeholders(cfg)

    assert replaced["solana"]["url"].endswith("api-key=mock_key")
    assert replaced["solana"]["nested"]["ws"].endswith("api-key=mock_key")
    assert replaced["list"][0] == "mock_key"
    assert replaced["list"][1]["deep"] == "value-mock_key"


def test_reload_config_replaces_placeholders(monkeypatch):
    main = _import_main()

    monkeypatch.setenv("HELIUS_KEY", "mock_key")

    async def fake_load_async():
        cfg = {
            "solana_scanner": {
                "helius_ws_url": "wss://mainnet.helius-rpc.com/?api-key=${HELIUS_KEY}",
                "api_keys": {"bitquery": "${HELIUS_KEY}"},
            },
            "risk": {
                "max_drawdown": 1.0,
                "stop_loss_pct": 0.0,
                "take_profit_pct": 0.0,
            },
        }
        return main.replace_placeholders(cfg), {"solana_scanner", "risk"}

    monkeypatch.setattr(main, "load_config_async", fake_load_async)

    config = {}
    ctx = main.BotContext({}, {}, {}, config)
    risk_manager = main.RiskManager(main.RiskConfig(1.0, 0.0, 0.0))
    rotator = types.SimpleNamespace(config={})
    guard = main.OpenPositionGuard(1)

    asyncio.run(
        main.reload_config(config, ctx, risk_manager, rotator, guard, force=True)
    )

    assert (
        config["solana_scanner"]["helius_ws_url"]
        == "wss://mainnet.helius-rpc.com/?api-key=mock_key"
    )
    assert config["solana_scanner"]["api_keys"]["bitquery"] == "mock_key"
