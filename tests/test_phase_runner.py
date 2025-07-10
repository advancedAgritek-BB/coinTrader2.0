import asyncio
from crypto_bot.phase_runner import PhaseRunner, BotContext


def test_phase_runner_executes_in_order_and_records_timings():
    calls = []

    async def phase_one(ctx):
        await asyncio.sleep(0.01)
        calls.append("phase_one")

    async def phase_two(ctx):
        await asyncio.sleep(0.01)
        calls.append("phase_two")

    async def phase_three(ctx):
        await asyncio.sleep(0.01)
        calls.append("phase_three")

    async def run_phases():
        runner = PhaseRunner([phase_one, phase_two, phase_three])
        ctx = BotContext(positions={}, df_cache={}, regime_cache={}, config={})
        timings = await runner.run(ctx)
        return timings

    timings = asyncio.run(run_phases())

    assert calls == ["phase_one", "phase_two", "phase_three"]
    assert set(timings.keys()) == {"phase_one", "phase_two", "phase_three"}
