import asyncio
import builtins
import time
from types import SimpleNamespace

import pandas as pd

import crypto_bot.console_control as console_control


def test_control_loop(monkeypatch):
    state = {"running": True, "reload": False}
    commands = ["stop", "start", "reload", "quit"]
    seen = []

    def fake_input(prompt=""):
        seen.append(state["running"])
        return commands.pop(0)

    monkeypatch.setattr(builtins, "input", fake_input)

    ctx = SimpleNamespace(active_universe=[], config={"timeframes": []})
    session_state = SimpleNamespace(df_cache={})
    asyncio.run(console_control.control_loop(state, ctx, session_state))

    # After first command ('stop') the state should have been False
    assert seen[1] is False
    # After second command ('start') the state should have been True
    assert seen[2] is True
    # Reload command should set the reload flag
    assert state["reload"] is True
    # Function should exit after processing all commands
    assert len(seen) == 4
    # "quit" stops the loop and leaves the bot stopped
    assert state["running"] is False


def test_control_loop_updates_state(monkeypatch):
    inputs = iter(["stop", "start", "reload", "quit"])

    def fake_input(prompt=""):
        return next(inputs)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(console_control.asyncio, "to_thread", fake_to_thread)

    state = {"running": True, "reload": False}
    ctx = SimpleNamespace(active_universe=[], config={"timeframes": []})
    session_state = SimpleNamespace(df_cache={})
    asyncio.run(console_control.control_loop(state, ctx, session_state))
    assert state["running"] is False


def test_reload_command(monkeypatch, tmp_path):
    inputs = iter(["reload", "quit"])

    def fake_input(prompt=""):
        return next(inputs)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(console_control.asyncio, "to_thread", fake_to_thread)

    import sys, types
    stub = types.ModuleType("crypto_bot.main")
    load_calls = []

    async def fake_reload(config, *_a, force=False, **_k):
        load_calls.append(True)
        config["reloaded"] = True
        state.pop("reload", None)

    stub.reload_config = fake_reload
    monkeypatch.setitem(sys.modules, "crypto_bot.main", stub)
    main = stub

    state = {"running": True}
    ctx = SimpleNamespace(active_universe=[], config={"timeframes": []})
    session_state = SimpleNamespace(df_cache={})
    asyncio.run(console_control.control_loop(state, ctx, session_state))

    # reload command should set the flag
    assert state["reload"] is True

    config = {}
    asyncio.run(main.reload_config(config, None, None, None, None, force=True))

    assert state.get("reload") is None
    assert load_calls and config.get("reloaded")


def test_panic_sell_command(monkeypatch):
    inputs = iter(["panic sell", "quit"])

    def fake_input(prompt=""):
        return next(inputs)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    printed = []

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(console_control.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr("builtins.print", lambda msg: printed.append(msg))

    state = {"running": True}
    ctx = SimpleNamespace(active_universe=[], config={"timeframes": []})
    session_state = SimpleNamespace(df_cache={})
    asyncio.run(console_control.control_loop(state, ctx, session_state))

    assert state.get("liquidate_all") is True
    assert any("Liquidation" in p for p in printed)


def test_control_loop_eof(monkeypatch, caplog):
    def fake_input(prompt=""):
        raise EOFError

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(console_control.asyncio, "to_thread", fake_to_thread)

    state = {"running": True}
    ctx = SimpleNamespace(active_universe=[], config={"timeframes": []})
    session_state = SimpleNamespace(df_cache={})
    with caplog.at_level("WARNING"):
        asyncio.run(console_control.control_loop(state, ctx, session_state))

    assert state["running"] is True
    assert any("EOF" in msg for msg in caplog.text.splitlines())


def test_status_command(monkeypatch, capsys):
    inputs = iter(["status", "quit"])

    def fake_input(prompt=""):
        return next(inputs)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(console_control.asyncio, "to_thread", fake_to_thread)

    df = pd.DataFrame([1], index=[pd.Timestamp("2024-01-01")])
    ctx = SimpleNamespace(active_universe=["BTC/USD"], config={"timeframes": ["1h"]})
    session_state = SimpleNamespace(df_cache={"1h": {"BTC/USD": df}})
    console_control.market_loader.failed_symbols["BTC/USD"] = {
        "time": time.time(),
        "delay": 60,
        "count": 1,
        "disabled": False,
    }
    state = {"running": True}
    asyncio.run(console_control.control_loop(state, ctx, session_state))
    out = capsys.readouterr().out.splitlines()
    assert any("BTC/USD" in line for line in out)
    assert any("1h" in line for line in out)
    console_control.market_loader.failed_symbols.clear()
