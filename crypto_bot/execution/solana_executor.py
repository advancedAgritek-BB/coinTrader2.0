"""Solana DEX execution helpers."""

from typing import Dict, Optional
import os
import json
import base64
import requests
from solana.rpc.api import Client
from solana.keypair import Keypair
from solana.transaction import Transaction

from crypto_bot.utils.telegram import send_message
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor


JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"


def execute_swap(
    token_in: str,
    token_out: str,
    amount: float,
    telegram_token: str,
    chat_id: str,
    dry_run: bool = True,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
    mempool_cfg: Optional[Dict] = None,
) -> Dict:
    """Execute a swap on Solana using the Jupiter aggregator."""

    msg = f"Swapping {amount} {token_in} to {token_out}"
    send_message(telegram_token, chat_id, msg)

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
        return result

    private_key = os.getenv("SOLANA_PRIVATE_KEY")
    if not private_key:
        raise ValueError("SOLANA_PRIVATE_KEY environment variable not set")

    keypair = Keypair.from_secret_key(bytes(json.loads(private_key)))
    rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    client = Client(rpc_url)

    quote_resp = requests.get(
        JUPITER_QUOTE_URL,
        params={
            "inputMint": token_in,
            "outputMint": token_out,
            "amount": int(amount),
            "slippageBps": 50,
        },
        timeout=10,
    )
    quote_resp.raise_for_status()
    route = quote_resp.json()["data"][0]

    swap_resp = requests.post(
        JUPITER_SWAP_URL,
        json={"route": route, "userPublicKey": str(keypair.public_key)},
        timeout=10,
    )
    swap_resp.raise_for_status()
    swap_tx = swap_resp.json()["swapTransaction"]

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
    return result
