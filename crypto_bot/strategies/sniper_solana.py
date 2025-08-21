import os
import asyncio
import time
from typing import Any, Dict, Optional
from loguru import logger

from crypto_bot.solana.helius_client import HeliusClient
from crypto_bot.solana.jupiter_client import JupiterClient
from crypto_bot.solana.raydium_client import RaydiumClient

# Simple config (could be moved to YAML)
SLIPPAGE_BPS = int(os.getenv("SNIPER_SLIPPAGE_BPS", "200"))     # 2%
MAX_PRICE_IMPACT_BPS = int(os.getenv("SNIPER_MAX_IMPACT_BPS", "250"))  # 2.5%
MIN_LP_USD = float(os.getenv("SNIPER_MIN_LP_USD", "10000"))
MIN_TOKEN_AGE_SEC = int(os.getenv("SNIPER_MIN_AGE_SEC", "60"))
BUY_AMOUNT_USD = float(os.getenv("SNIPER_BUY_USD", "50"))

USDC_MINT = os.getenv("USDC_MINT", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
SOL_MINT  = os.getenv("SOL_MINT",  "So11111111111111111111111111111111111111112")


class Strategy:
    """
    Wraps an on-chain runner. The evaluation engine will instantiate this Strategy,
    but there is no OHLCV 'evaluate'. We spawn our own task in start() and stop it on shutdown.
    """

    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._runner(), name="sniper-solana-runner")
        logger.info("sniper_solana runner started.")

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        logger.info("sniper_solana runner stopped.")

    # --- POLICY / SCORING ---

    def _passes_risk_gates(self, meta: Dict[str, Any], pool_info: Dict[str, Any]) -> bool:
        """
        Very conservative gates: min LP USD, token age, no obvious mint authority risks.
        Expand with renounced checks, freeze authority checks (requires RPC read).
        """
        age_ok = (meta.get("createdAt", 0) or 0) <= (int(time.time()) - MIN_TOKEN_AGE_SEC)
        lp_ok = float(pool_info.get("liquidityUsd", 0) or 0) >= MIN_LP_USD
        return age_ok and lp_ok

    # --- RUNNER ---

    async def _runner(self):
        """
        Poll-based example (replace with Helius webhook/WebSocket if you prefer).
        For brevity, demonstrate a simple loop that:
          - discovers candidate mints from a source you already have in repo (or Raydium)
          - fetches metadata via Helius
          - quotes and tries to build a swap via Jupiter (fallback Raydium)
        """
        async with HeliusClient() as helius, JupiterClient() as jup, RaydiumClient() as ray:
            # You likely already have a scanner in your repo; worst case use Raydium new pools:
            async def discover_new_mints() -> list[str]:
                # TODO: wire to your scanner; this is a stub returning []
                return []

            while not self._stop.is_set():
                try:
                    mints = await discover_new_mints()
                    if not mints:
                        await asyncio.sleep(2.0)
                        continue

                    meta_map = await helius.get_token_metadata(mints)
                    for mint in mints:
                        meta = meta_map.get(mint) or {}
                        # Minimal pool info (if you have your own source, use that data instead)
                        pools = await ray.pools_by_mint(mint)
                        pool = pools[0] if pools else {}
                        if not pool or not self._passes_risk_gates(meta, pool):
                            continue

                        # Attempt Jupiter route USDC -> mint
                        # For demo purposes, compute a nominal USDC amount in base units (6 decimals)
                        usdc_amount = int(BUY_AMOUNT_USD * 1_000_000)
                        route = await jup.quote(USDC_MINT, mint, usdc_amount, SLIPPAGE_BPS)
                        if not route:
                            # Fallback or skip; you can build a Raydium Tx here if desired
                            continue

                        # (Optional) check price impact in bps if route has that info
                        # if route.get("priceImpactPct", 0) * 10000 > MAX_PRICE_IMPACT_BPS: continue

                        # At this point call your wallet/tx builder to submit the trade using
                        # Jupiter swap instructions or your own Raydium instruction builder.
                        # This repo-specific part isn’t shown here because your signer stack
                        # and account abstraction isn’t in the logs. Hook here:
                        logger.info(f"[SNIPER] would buy mint {mint} via Jupiter route id={route.get('id')}")

                    await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.exception(f"sniper runner error: {e!r}")
                    await asyncio.sleep(1.0)
