from __future__ import annotations

"""Adjust solana_scanner.max_tokens_per_scan based on BTC balance."""

import asyncio
from pathlib import Path
import yaml
import os
import ccxt.async_support as ccxt

from crypto_bot.utils.symbol_utils import fix_symbol

CONFIG_PATH = Path(__file__).resolve().parents[1] / "crypto_bot" / "config.yaml"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    strat_dir = CONFIG_PATH.parent.parent / "config" / "strategies"
    trend_file = strat_dir / "trend_bot.yaml"
    if trend_file.exists():
        with open(trend_file) as sf:
            overrides = yaml.safe_load(sf) or {}
        trend_cfg = data.get("trend_bot", {})
        if isinstance(trend_cfg, dict):
            trend_cfg.update(overrides)
        else:
            trend_cfg = overrides
        data["trend_bot"] = trend_cfg
    if "symbol" in data:
        data["symbol"] = fix_symbol(data["symbol"])
    if "symbols" in data:
        data["symbols"] = [fix_symbol(s) for s in data.get("symbols", [])]

    trading_cfg = data.get("trading", {}) or {}
    raw_ex = data.get("exchange") or trading_cfg.get("exchange") or os.getenv("EXCHANGE")
    if isinstance(raw_ex, dict):
        ex_cfg = dict(raw_ex)
    else:
        ex_cfg = {"name": raw_ex}
    ex_cfg.setdefault("name", "kraken")
    ex_cfg.setdefault("max_concurrency", 3)
    ex_cfg.setdefault("request_timeout_ms", 10000)
    data["exchange"] = ex_cfg

    return data


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
