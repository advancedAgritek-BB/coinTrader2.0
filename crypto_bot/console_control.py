import asyncio
from typing import Dict, Any

async def control_loop(state: Dict[str, Any]) -> None:
    """Simple console loop to start/stop the bot via user input.

    This coroutine reads commands from ``input`` and updates ``state``.
    Recognised commands:
    - ``start``: set ``state['running'] = True``
    - ``stop``: set ``state['running'] = False``
    - ``quit``: exit the loop
    """
    while True:
        # running input in a thread avoids blocking the event loop
        cmd = await asyncio.to_thread(input, "")
        cmd = cmd.strip().lower()
        if cmd == "start":
            state["running"] = True
        elif cmd == "stop":
            state["running"] = False
        elif cmd == "quit":
            break

