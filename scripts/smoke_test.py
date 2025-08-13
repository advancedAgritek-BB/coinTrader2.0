import asyncio
import sys
from pathlib import Path

import yaml
import os

# Ensure repository root is on sys.path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Avoid interactive prompts for API credentials
os.environ.setdefault("PYTEST_CURRENT_TEST", "1")

from crypto_bot.execution.cex_executor import get_exchange
from crypto_bot.utils.market_loader import load_kraken_symbols, update_ohlcv_cache


async def main() -> None:
    with open("config.example.testing.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    exchange, ws_client = get_exchange(config)

    subset: list[str] = []
    try:
        symbols = await load_kraken_symbols(exchange, config=config) or []
        subset = symbols[:5]
        print(f"Smoke: symbols={len(subset)}")
    except Exception as exc:  # pragma: no cover - network issues
        print(f"Smoke: symbols=0 (error: {exc.__class__.__name__})")

    if subset:
        cache: dict = {}
        tf = (config.get("timeframes") or ["1h"])[0]
        try:
            await update_ohlcv_cache(
                exchange,
                cache,
                subset[:2],
                timeframe=tf,
                limit=2,
                config=config,
            )
        except Exception:  # pragma: no cover - best effort
            pass

    if hasattr(exchange, "close"):
        if asyncio.iscoroutinefunction(getattr(exchange, "close", None)):
            await exchange.close()
        else:
            await asyncio.to_thread(exchange.close)
    if ws_client and hasattr(ws_client, "close"):
        await ws_client.close()


if __name__ == "__main__":
    asyncio.run(main())
