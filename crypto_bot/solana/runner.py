from __future__ import annotations

"""Asynchronous runner for Solana meme-wave sniping."""

import asyncio
import logging
from typing import Mapping

from .watcher import PoolWatcher, NewPoolEvent
from .score import score_event
from . import executor
from crypto_bot.solana_trading import cross_chain_trade

logger = logging.getLogger(__name__)


async def run(config: Mapping[str, object]) -> None:
    """Run the meme-wave sniper loop using ``config`` options with optional
    cross-chain arbitrage."""

    pool_cfg = config.get("pool", {}) if isinstance(config, Mapping) else {}
    url = str(pool_cfg.get("url", ""))
    interval = float(pool_cfg.get("interval", 5))
    ws_url = pool_cfg.get("websocket_url")
    program_id = pool_cfg.get(
        "raydium_program_id",
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    )
    ml_filter = bool(pool_cfg.get("ml_filter", False))
    watcher = PoolWatcher(url, interval, ws_url, program_id, ml_filter=ml_filter)

    arb_cfg = config.get("arbitrage", {}) if isinstance(config, Mapping) else {}

    try:
        async for event in watcher.watch():
            score = score_event(event, config.get("scoring", {}))
            if score >= 0.7:
                await executor.snipe(event, score, config.get("execution", {}))
                if watcher._predict_breakout(event) >= 0.7 and arb_cfg:
                    try:
                        await cross_chain_trade(
                            arb_cfg.get("exchange"),
                            arb_cfg.get("ws_client"),
                            str(arb_cfg.get("symbol", "")),
                            str(arb_cfg.get("side", "buy")),
                            float(arb_cfg.get("amount", 0)),
                            dry_run=bool(arb_cfg.get("dry_run", True)),
                            slippage_bps=int(arb_cfg.get("slippage_bps", 50)),
                            use_websocket=bool(arb_cfg.get("use_websocket", False)),
                            notifier=arb_cfg.get("notifier"),
                            mempool_monitor=arb_cfg.get("mempool_monitor"),
                            mempool_cfg=arb_cfg.get("mempool_cfg"),
                            config=config,
                        )
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.error("Arbitrage failed: %s", exc)
    except asyncio.CancelledError:
        watcher.stop()
        raise
    except Exception as e:  # pragma: no cover - best effort
        logger.error("Runner error: %s", e)
