import asyncio
from crypto_bot import console_control


def test_control_loop_updates_state(monkeypatch):
    inputs = iter(["stop", "start", "quit"])

    def fake_input(prompt=""):
        return next(inputs)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(console_control.asyncio, "to_thread", fake_to_thread)

    state = {"running": True}
    asyncio.run(console_control.control_loop(state))
    assert state["running"] is False
