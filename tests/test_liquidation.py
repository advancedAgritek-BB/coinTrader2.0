import asyncio
import crypto_bot.main as main

class DummyCtx:
    pass

def test_liquidate_state_calls_force_exit(monkeypatch):
    state = {"liquidate_all": True}
    called = {}

    async def fake_force_exit_all(ctx):
        called["hit"] = True

    monkeypatch.setattr(main, "force_exit_all", fake_force_exit_all)

    async def run():
        if state.get("liquidate_all"):
            await main.force_exit_all(DummyCtx())
            state["liquidate_all"] = False

    asyncio.run(run())

    assert called.get("hit") is True
    assert state["liquidate_all"] is False
