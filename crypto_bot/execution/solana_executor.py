"""Solana DEX execution helpers."""

from typing import Dict, Optional
import os
import json
import base64
import aiohttp

from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.utils.notifier import Notifier
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
from crypto_bot import tax_logger
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from pathlib import Path


logger = setup_logger(__name__, LOG_DIR / "execution.log")


JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"
JITO_BUNDLE_URL = "https://mainnet.block-engine.jito.wtf/api/v1/bundles"


async def execute_swap(
    token_in: str,
    token_out: str,
    amount: float,
    telegram_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    notifier: Optional[TelegramNotifier] = None,
    slippage_bps: int = 50,
    dry_run: bool = True,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
    mempool_cfg: Optional[Dict] = None,
    config: Optional[Dict] = None,
    jito_key: Optional[str] = None,
) -> Dict:
    """Execute a swap on Solana using the Jupiter aggregator."""

    if notifier is None:
        if telegram_token is None or chat_id is None:
            raise ValueError("telegram_token/chat_id or notifier must be provided")
        notifier = Notifier(telegram_token, chat_id)

    msg = f"Swapping {amount} {token_in} to {token_out}"
    err = notifier.notify(msg)
    if err:
        logger.error("Failed to send message: %s", err)

    config = config or {}

    cfg = mempool_cfg or {}
    if mempool_monitor and cfg.get("enabled"):
        threshold = cfg.get("suspicious_fee_threshold", 0.0)
        action = cfg.get("action", "pause")
        if mempool_monitor.is_suspicious(threshold):
            err_msg = notifier.notify("High priority fees detected")
            if err_msg:
                logger.error("Failed to send message: %s", err_msg)
            if action == "pause":
                return {
                    "token_in": token_in,
                    "token_out": token_out,
                    "amount": amount,
                    "paused": True,
                }
            if action == "reprice":
                amount *= cfg.get("reprice_multiplier", 1.0)
        fee = mempool_monitor.fetch_priority_fee()
        gas_limit = config.get("gas_threshold_gwei", 0.0)
        if gas_limit and fee > gas_limit:
            logger.warning("Swap aborted due to high priority fee: %s", fee)
            return {}
        tp = config.get("take_profit_pct") or config.get("risk", {}).get("take_profit_pct", 0.0)
        if not gas_limit and tp and fee > tp * 0.05:
            logger.warning("Swap aborted due to high priority fee: %s", fee)
            return {}

    if dry_run:
        tx_hash = "DRYRUN"
        result = {
            "token_in": token_in,
            "token_out": token_out,
            "amount": amount,
            "tx_hash": tx_hash,
        }
        err_res = notifier.notify(f"Swap executed: {result}")
        if err_res:
            logger.error("Failed to send message: %s", err_res)
        logger.info(
            "Swap completed: %s -> %s amount=%s tx=%s",
            token_in,
            token_out,
            amount,
            tx_hash,
        )
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
    rpc_url = os.getenv(
        "SOLANA_RPC_URL",
        f"https://mainnet.helius-rpc.com/?api-key={os.getenv('HELIUS_KEY', '')}",
    )
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
        if not quote_data.get("data"):
            logger.warning("No routes returned from Jupiter")
            return {}
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
                logger.info(
                    "Swap skipped: %s -> %s amount=%s due to slippage",
                    token_in,
                    token_out,
                    amount,
                )
                err_skip = notifier.notify("Trade skipped due to slippage.")
                if err_skip:
                    logger.error("Failed to send message: %s", err_skip)
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

    if jito_key is None:
        jito_key = os.getenv("JITO_KEY")

    if jito_key:
        signed_tx = base64.b64encode(tx.serialize()).decode()
        async with aiohttp.ClientSession() as jito_session:
            async with jito_session.post(
                JITO_BUNDLE_URL,
                json={"transactions": [signed_tx]},
                headers={"Authorization": f"Bearer {jito_key}"},
                timeout=10,
            ) as bundle_resp:
                bundle_resp.raise_for_status()
                bundle_data = await bundle_resp.json()
        tx_hash = bundle_data.get("signature") or bundle_data.get("bundleId")
    else:
        send_res = client.send_transaction(tx, keypair)
        tx_hash = send_res["result"]

    result = {
        "token_in": token_in,
        "token_out": token_out,
        "amount": amount,
        "tx_hash": tx_hash,
        "route": route,
    }
    err = notifier.notify(f"Swap executed: {result}")
    if err:
        logger.error("Failed to send message: %s", err)
    logger.info(
        "Swap completed: %s -> %s amount=%s tx=%s",
        token_in,
        token_out,
        amount,
        tx_hash,
    )
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
