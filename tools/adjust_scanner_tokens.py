from __future__ import annotations

import asyncio
from pathlib import Path
import yaml
import ccxt.async_support as ccxt

from configy import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "crypto_bot" / "config.yaml"


async def fetch_btc_balance(exchange) -> float:
    if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
        bal = await exchange.fetch_balance()
    else:
        bal = await asyncio.to_thread(exchange.fetch_balance)
    val = bal.get("BTC", {})
    return float(val.get("free", val if isinstance(val, (int, float)) else 0))


async def adjust(config: dict) -> None:
    ex = config.get("exchange", "kraken")
    if isinstance(ex, dict):
        name = ex.get("name", "kraken").lower()
        params = {"enableRateLimit": True}
        timeout = ex.get("request_timeout_ms")
        if timeout:
            params["timeout"] = int(timeout)
        max_conc = ex.get("max_concurrency")
    else:
        name = str(ex).lower()
        params = {"enableRateLimit": True}
        max_conc = None
    exchange_cls = getattr(ccxt, name)
    exchange = exchange_cls(params)
    if max_conc is not None:
        setattr(exchange, "max_concurrency", int(max_conc))
    balance = 0.0
    try:
        balance = await fetch_btc_balance(exchange)
    finally:
        await exchange.close()

    if balance >= 1:
        max_tokens = 20
    elif balance >= 0.5:
        max_tokens = 10
    else:
        max_tokens = 5
    config.setdefault("solana_scanner", {})["max_tokens_per_scan"] = max_tokens
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)
    print(f"Set max_tokens_per_scan to {max_tokens} (BTC balance: {balance:.4f})")


def main() -> None:
    cfg = load_config()
    asyncio.run(adjust(cfg))


if __name__ == "__main__":
    main()
