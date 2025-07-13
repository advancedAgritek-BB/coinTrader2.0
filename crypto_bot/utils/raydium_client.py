from __future__ import annotations

import os
import json
import base64
from typing import Any, Mapping

import aiohttp
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
from solders.transaction import VersionedTransaction
from solders.signature import Signature
from solders.keypair import Keypair

from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "raydium_client.log")

QUOTE_URL = "https://transaction-v1.raydium.io/compute/swap-base-in"
TX_URL = "https://transaction-v1.raydium.io/transaction/swap-base-in"


def get_wallet() -> Keypair:
    """Return wallet keypair from ``SOLANA_PRIVATE_KEY``."""
    key = os.getenv("SOLANA_PRIVATE_KEY")
    if not key:
        raise ValueError("SOLANA_PRIVATE_KEY not set")
    secret = bytes(json.loads(key))
    return Keypair.from_bytes(secret)


async def get_swap_quote(
    input_mint: str,
    output_mint: str,
    amount: int,
    slippage_bps: int = 50,
    tx_version: str = "V0",
) -> Mapping[str, Any]:
    """Return Raydium swap quote json."""
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": amount,
        "slippageBps": slippage_bps,
        "txVersion": tx_version,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(QUOTE_URL, params=params, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json()
    logger.info("Fetched quote %s", data.get("id"))
    return data


async def execute_swap(
    wallet_address: str,
    input_account: str,
    output_account: str,
    swap_response: Mapping[str, Any],
    *,
    wrap_sol: bool = True,
    unwrap_sol: bool = False,
    compute_unit_price: int = 1000,
    tx_version: str = "V0",
    risk_manager: object | None = None,
) -> Mapping[str, Any]:
    """Execute a Raydium swap and return the RPC result."""
    payload = dict(swap_response)
    payload.update(
        {
            "walletAddress": wallet_address,
            "inputAccount": input_account,
            "outputAccount": output_account,
            "wrapAndUnwrapSol": {"wrapSol": wrap_sol, "unwrapSol": unwrap_sol},
            "computeUnitPriceMicroLamports": compute_unit_price,
            "txVersion": tx_version,
        }
    )
    async with aiohttp.ClientSession() as session:
        async with session.post(TX_URL, json=payload, timeout=10) as resp:
            resp.raise_for_status()
            tx_data = await resp.json()

    tx_b64 = tx_data.get("swapTransaction") or tx_data.get("transaction")
    if not tx_b64:
        logger.error("Transaction missing from response")
        return tx_data
    raw = base64.b64decode(tx_b64)
    kp = get_wallet()
    if tx_version == "V0":
        vt = VersionedTransaction.from_bytes(raw)
        vt = VersionedTransaction(vt.message, [kp])
        send_bytes = bytes(vt)
    else:
        tx = Transaction.deserialize(raw)
        tx.sign(kp)
        send_bytes = tx.serialize()

    rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    async with AsyncClient(rpc_url) as client:
        res = await client.send_raw_transaction(send_bytes)
        sig = getattr(res, "value", None) or res.get("result")
        if isinstance(sig, str):
            sig_obj = Signature.from_string(sig)
        else:
            sig_obj = sig
        if sig_obj is not None:
            await client.confirm_transaction(sig_obj)
            tx_hash = str(sig_obj)
        else:
            tx_hash = ""
    result = {"tx_hash": tx_hash, "response": tx_data}
    logger.info("Swap executed tx=%s", tx_hash)
    return result


async def sniper_trade(
    input_mint: str,
    output_mint: str,
    amount: int,
    notifier: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Execute a simple snipe trade and convert profits to BTC."""
    from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
    from crypto_bot.fund_manager import auto_convert_funds

    cfg = config or {}
    quote = await get_swap_quote(
        input_mint,
        output_mint,
        amount,
        tx_version=cfg.get("tx_version", "V0"),
    )

    risk_cfg = RiskConfig(max_drawdown=1.0, stop_loss_pct=0.01, take_profit_pct=0.01)
    rm = RiskManager(risk_cfg)
    size = rm.position_size(1.0, float(amount))

    swap_res = await execute_swap(
        cfg.get("wallet_address", ""),
        input_mint,
        output_mint,
        quote.get("data", quote),
        tx_version=cfg.get("tx_version", "V0"),
        risk_manager=rm,
    )

    await auto_convert_funds(
        cfg.get("wallet_address", ""),
        output_mint,
        "BTC",
        size,
        dry_run=True,
        notifier=notifier,
    )
    return swap_res
