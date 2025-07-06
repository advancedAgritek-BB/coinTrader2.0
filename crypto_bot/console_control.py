import asyncio

async def control_loop(state: dict) -> None:
    """Simple console control loop to pause or stop the bot."""
    while True:
        try:
            cmd = await asyncio.to_thread(input, "")
        except EOFError:
            break
        if cmd is None:
            continue
        cmd = cmd.strip().lower()
        if cmd in {"stop", "pause"}:
            state["running"] = False
            print("Bot paused.")
        elif cmd in {"start", "resume"}:
            state["running"] = True
            print("Bot resumed.")
        elif cmd in {"quit", "exit"}:
            state["running"] = False
            break
