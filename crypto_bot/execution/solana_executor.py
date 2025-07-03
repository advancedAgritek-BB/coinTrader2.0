"""Solana DEX execution helpers."""

from typing import Dict, Optional
import os
import json
import base64
import aiohttp

from crypto_bot.utils.telegram import send_message
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
from crypto_bot import tax_logger
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/execution.log")


JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"


async def execute_swap(
    token_in: str,
    token_out: str,
    amount: float,
    telegram_token: str,
    chat_id: str,
    slippage_bps: int = 50,
    dry_run: bool = True,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
    mempool_cfg: Optional[Dict] = None,
    config: Optional[Dict] = None,
) -> Dict:
    """Execute a swap on Solana using the Jupiter aggregator."""

    msg = f"Swapping {amount} {token_in} to {token_out}"
    send_message(telegram_token, chat_id, msg)

    config = config or {}

    cfg = mempool_cfg or {}
    if mempool_monitor and cfg.get("enabled"):
        threshold = cfg.get("suspicious_fee_threshold", 0.0)
        action = cfg.get("action", "pause")
        if mempool_monitor.is_suspicious(threshold):
            send_message(telegram_token, chat_id, "High priority fees detected")
            if action == "pause":
                return {
                    "token_in": token_in,
                    "token_out": token_out,
                    "amount": amount,
                    "paused": True,
                }
            if action == "reprice":
                amount *= cfg.get("reprice_multiplier", 1.0)

    if dry_run:
        tx_hash = "DRYRUN"
        result = {
            "token_in": token_in,
            "token_out": token_out,
            "amount": amount,
            "tx_hash": tx_hash,
        }
        send_message(telegram_token, chat_id, f"Swap executed: {result}")
        logger.info(
            "Swap executed - tx=%s in=%s out=%s amount=%s dry_run=%s",
            tx_hash,
            token_in,
            token_out,
            amount,
            True,
        )
        if (config or {}).get("tax_tracking", {}).get("enabled"):
            try:
                tax_logger.record_exit({"symbol": token_in, "amount": amount, "side": "sell"})
                tax_logger.record_entry({"symbol": token_out, "amount": amount, "side": "buy"})
            except Exception:
                pass
        return result

    from solana.rpc.api import Client
    from solana.keypair import Keypair
    from solana.transaction import Transaction

    private_key = os.getenv("SOLANA_PRIVATE_KEY")
    if not private_key:
        raise ValueError("SOLANA_PRIVATE_KEY environment variable not set")

    keypair = Keypair.from_secret_key(bytes(json.loads(private_key)))
    rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    client = Client(rpc_url)

    async with aiohttp.ClientSession() as session:
        async with session.get(
            JUPITER_QUOTE_URL,
            params={
                "inputMint": token_in,
                "outputMint": token_out,
                "amount": int(amount),
                "slippageBps": slippage_bps,
            },
            timeout=10,
        ) as quote_resp:
            quote_resp.raise_for_status()
            quote_data = await quote_resp.json()
        route = quote_data["data"][0]

        try:
            async with session.get(
                JUPITER_QUOTE_URL,
                params={
                    "inputMint": token_out,
                    "outputMint": token_in,
                    "amount": int(route["outAmount"]),
                    "slippageBps": slippage_bps,
                },
                timeout=10,
            ) as back_resp:
                back_resp.raise_for_status()
                back_data = await back_resp.json()
            back_route = back_data["data"][0]
            ask = float(route["outAmount"]) / float(route["inAmount"])
            bid = float(back_route["inAmount"]) / float(back_route["outAmount"])
            slippage = (ask - bid) / ((ask + bid) / 2)
            if slippage > config.get("max_slippage_pct", 1.0):
                logger.warning("Trade skipped due to slippage.")
                send_message(telegram_token, chat_id, "Trade skipped due to slippage.")
                return {}
        except Exception as err:  # pragma: no cover - network
            logger.warning("Slippage check failed: %s", err)

        async with session.post(
            JUPITER_SWAP_URL,
            json={"route": route, "userPublicKey": str(keypair.public_key)},
            timeout=10,
        ) as swap_resp:
            swap_resp.raise_for_status()
            swap_data = await swap_resp.json()
        swap_tx = swap_data["swapTransaction"]

    raw_tx = base64.b64decode(swap_tx)
    tx = Transaction.deserialize(raw_tx)
    tx.sign(keypair)
    send_res = client.send_transaction(tx, keypair)
    tx_hash = send_res["result"]

    result = {
        "token_in": token_in,
        "token_out": token_out,
        "amount": amount,
        "tx_hash": tx_hash,
    }
    send_message(telegram_token, chat_id, f"Swap executed: {result}")
    logger.info(
        "Swap executed - tx=%s in=%s out=%s amount=%s dry_run=%s",
        tx_hash,
        token_in,
        token_out,
        amount,
        False,
    )
    if (config or {}).get("tax_tracking", {}).get("enabled"):
        try:
            tax_logger.record_exit({"symbol": token_in, "amount": amount, "side": "sell"})
            tax_logger.record_entry({"symbol": token_out, "amount": amount, "side": "buy"})
        except Exception:
            pass
    return result
