"""Utilities for replaying Solana pool history."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Iterable, List

import pandas as pd
import requests

from crypto_bot.solana.watcher import NewPoolEvent
from crypto_bot.solana.sniper_solana import score_new_pool, generate_signal
from crypto_bot.solana.risk import RiskTracker


__all__ = ["fetch_pool_history", "main"]


def _to_timestamp(t: str | int | float | datetime) -> int:
    """Convert ``t`` to a unix timestamp (seconds)."""
    return int(pd.to_datetime(t, utc=True).timestamp())


def fetch_pool_history(start: str | int | float | datetime, end: str | int | float | datetime, rpc_url: str) -> List[NewPoolEvent]:
    """Return a list of :class:`NewPoolEvent` objects between ``start`` and ``end``.

    Parameters
    ----------
    start, end:
        Datetime-like boundaries for the query.
    rpc_url:
        QuickNode RPC URL supporting ``dex.getNewPools``.
    """

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "dex.getNewPools",
        "params": {
            "protocols": ["raydium"],
            "startTime": _to_timestamp(start),
            "endTime": _to_timestamp(end),
            "limit": 1000,
        },
    }

    resp = requests.post(rpc_url, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    result = data.get("result", data)
    pools = result.get("pools") or result.get("data") or result

    events: List[NewPoolEvent] = []
    for item in pools:
        events.append(
            NewPoolEvent(
                pool_address=item.get("address")
                or item.get("poolAddress")
                or item.get("pool_address")
                or "",
                token_mint=item.get("tokenMint") or item.get("token_mint") or "",
                creator=item.get("creator", ""),
                liquidity=float(item.get("liquidity", 0.0)),
                tx_count=int(item.get("txCount", item.get("tx_count", 0))),
                freeze_authority=item.get("freezeAuthority")
                or item.get("freeze_authority")
                or "",
                mint_authority=item.get("mintAuthority")
                or item.get("mint_authority")
                or "",
                timestamp=float(item.get("timestamp", 0.0)),
            )
        )
    return events


def main(argv: Iterable[str] | None = None) -> None:
    """Command line utility for simple sniper backtests."""

    p = argparse.ArgumentParser(description="Replay historical pool events")
    p.add_argument("--start", required=True, help="Start time (ISO or epoch)")
    p.add_argument("--end", required=True, help="End time (ISO or epoch)")
    p.add_argument("--rpc-url", required=True, help="QuickNode RPC endpoint")
    p.add_argument("--config", help="YAML config with sniper settings")
    p.add_argument(
        "--risk-state",
        default="risk_state.json",
        help="Path for RiskTracker persistence",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    events = fetch_pool_history(args.start, args.end, args.rpc_url)
    cfg = {}
    if args.config:
        import yaml

        with open(args.config) as fh:
            cfg = yaml.safe_load(fh) or {}

    tracker = RiskTracker(args.risk_state)

    for evt in events:
        score, direction = score_new_pool(evt, cfg, tracker)
        df = pd.DataFrame(
            {
                "open": [1.0, 1.02],
                "high": [1.01, 1.05],
                "low": [0.99, 1.0],
                "close": [1.0, 1.04],
                "volume": [1, 1],
            }
        )
        sig_score, sig_dir = generate_signal(
            df, {"token": evt.token_mint}, timeframe=cfg.get("timeframe")
        )
        print(
            f"{evt.pool_address} score={score:.2f} dir={direction} "
            f"signal={sig_dir} ({sig_score:.2f})"
        )


if __name__ == "__main__":  # pragma: no cover - manual use
    main()
