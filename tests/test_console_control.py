import asyncio
import builtins
import crypto_bot.console_control as console_control


def test_control_loop(monkeypatch):
    state = {"running": True, "reload": False}
    commands = ["stop", "start", "reload", "quit"]
    seen = []

    def fake_input(prompt=""):
        seen.append(state["running"])
        return commands.pop(0)

    monkeypatch.setattr(builtins, "input", fake_input)

    asyncio.run(console_control.control_loop(state))

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
    asyncio.run(console_control.control_loop(state))
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

    def fake_load():
        load_calls.append(True)
        return {}

    def maybe_reload_config(state, config):
        if state.get("reload"):
            cfg = fake_load()
            config.clear()
            config.update(cfg)
            state.pop("reload", None)

    stub.load_config = fake_load
    stub.maybe_reload_config = maybe_reload_config
    monkeypatch.setitem(sys.modules, "crypto_bot.main", stub)
    main = stub

    state = {"running": True}
    asyncio.run(console_control.control_loop(state))

    # reload command should set the flag
    assert state["reload"] is True

    config = {}
    main.maybe_reload_config(state, config)

    assert state.get("reload") is None
    assert load_calls
