import asyncio
import builtins
import crypto_bot.console_control as console_control


def test_control_loop(monkeypatch):
    state = {"running": True}
    commands = ["stop", "start", "quit"]
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
    # Function should exit after processing all commands
    assert len(seen) == 3
    assert state["running"] is True
