import asyncio

async def run_pipeline(config):
    from .scoring import scoring_loop
    from .execution import execution_loop

    await asyncio.gather(
        scoring_loop(config),
        execution_loop(config),
    )
