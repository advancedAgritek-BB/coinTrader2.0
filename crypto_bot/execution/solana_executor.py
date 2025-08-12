"""Solana DEX execution helpers."""

from typing import Dict, Optional
import os
import json
import base64
import asyncio
import sys
import aiohttp

try:  # pragma: no cover - optional dependency
    import keyring  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    keyring = None  # type: ignore

import crypto_bot.utils.telegram  # ensure submodule attribute for monkeypatch
from crypto_bot.utils.telegram import TelegramNotifier
from crypto_bot.utils.notifier import Notifier
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
from crypto_bot import tax_logger
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.token_registry import fetch_from_jupiter, get_decimals, to_base_units


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
    max_retries: int = 3,
) -> Dict:
    """Execute a swap on Solana using the Jupiter aggregator.

    Parameters
    ----------
    max_retries:
        Number of attempts when network calls fail. Defaults to ``3``.
    """

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
        if await mempool_monitor.is_suspicious(threshold):
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
        fee = await mempool_monitor.fetch_priority_fee()
        fee_cap = config.get("priority_fee_cap_micro_lamports", 0.0)
        if fee_cap and fee > fee_cap:
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

    decimals = await get_decimals(token_in)
    if decimals == 0:
        try:
            await fetch_from_jupiter()
        except Exception as exc:
            logger.error("Failed to refresh token decimals: %s", exc)
        decimals = await get_decimals(token_in)
        if decimals == 0:
            msg = f"Unknown token decimals for {token_in}"
            logger.warning(msg)
            err = notifier.notify(msg)
            if err:
                logger.error("Failed to send message: %s", err)
            return {}
    amount_base = to_base_units(amount, decimals)

    from solana.keypair import Keypair
    from solana.transaction import Transaction

    private_key = os.getenv("SOLANA_PRIVATE_KEY")
    if not private_key and keyring:
        try:
            private_key = keyring.get_password("solana", "private_key")
        except Exception:
            private_key = None
    if not private_key:
        raise ValueError("SOLANA_PRIVATE_KEY environment variable not set")

    keypair = Keypair.from_secret_key(bytes(json.loads(private_key)))
    rpc_url = os.getenv(
        "SOLANA_RPC_URL",
        f"https://mainnet.helius-rpc.com/v1/?api-key={os.getenv('HELIUS_KEY', '')}",
    )

    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            try:
                async with session.get(
                    JUPITER_QUOTE_URL,
                    params={
                        "inputMint": token_in,
                        "outputMint": token_out,
                        "amount": amount_base,
                        "slippageBps": slippage_bps,
                    },
                    timeout=10,
                ) as quote_resp:
                    quote_resp.raise_for_status()
                    quote_data = await quote_resp.json()
                break
            except aiohttp.ClientError as exc:
                if attempt >= max_retries - 1:
                    err_msg = notifier.notify(f"Quote failed: {exc}")
                    if err_msg:
                        logger.error("Failed to send message: %s", err_msg)
                    return {}
                await asyncio.sleep(1)
            except Exception as err:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                err_msg = notifier.notify(f"Quote error: {err}")
                if err_msg:
                    logger.error("Failed to send message: %s", err_msg)
                return {}
        if not quote_data.get("data"):
            logger.warning("No routes returned from Jupiter")
            return {}
        route = quote_data["data"][0]

        # Abort if available liquidity is too low
        liq = route.get("liquidity")
        if liq is None:
            try:
                liq = (route.get("marketInfos") or [{}])[0].get("liquidity")
            except Exception:
                liq = None
        if liq is not None:
            max_use = amount * config.get("max_liquidity_usage", 0.8)
            if liq < max_use:
                logger.warning("Swap aborted due to low liquidity: %s", liq)
                err = notifier.notify("Swap aborted: insufficient liquidity")
                if err:
                    logger.error("Failed to send message: %s", err)
                return {}

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
            try:
                ask = float(route["outAmount"]) / float(route["inAmount"])
                bid = float(back_route["inAmount"]) / float(back_route["outAmount"])
            except (KeyError, ValueError, ZeroDivisionError) as e:
                logger.warning("Slippage calc failed: %s - skipping check", e)
                slippage = 0.0
            else:
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
            logger.warning(
                "Slippage check failed for %s->%s amount=%s: %s",
                token_in,
                token_out,
                amount,
                err,
                exc_info=True,
            )

        confirm_exec = (config or {}).get("confirm_execution")
        if confirm_exec is None:
            if sys.stdin.isatty():
                try:
                    confirm_exec = input("Execute swap? [y/N]: ").strip().lower() in ("y", "yes")
                except Exception:
                    confirm_exec = False
            else:
                confirm_exec = False
        if not confirm_exec:
            logger.info("Swap aborted by user")
            return {}

        for attempt in range(max_retries):
            try:
                async with session.post(
                    JUPITER_SWAP_URL,
                    json={"route": route, "userPublicKey": str(keypair.public_key)},
                    timeout=10,
                ) as swap_resp:
                    swap_resp.raise_for_status()
                    swap_data = await swap_resp.json()
                break
            except Exception as err:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                err_msg = notifier.notify(f"Swap failed: {err}")
                if err_msg:
                    logger.error("Failed to send message: %s", err_msg)
                return {}
        swap_tx = swap_data["swapTransaction"]

    try:
        raw_tx = base64.b64decode(swap_tx)
    except Exception:
        raw_tx = swap_tx.encode()
    tx = Transaction.deserialize(raw_tx)
    tx.sign(keypair)

    serialize_fn = getattr(tx, "serialize", None)
    signed_bytes = serialize_fn() if callable(serialize_fn) else b""

    if jito_key is None:
        jito_key = os.getenv("JITO_KEY")

    from solana.rpc.async_api import AsyncClient
    from crypto_bot.solana import api_helpers

    tx_hash: Optional[str] = None
    async with AsyncClient(rpc_url) as client:
        if jito_key:
            try:
                signed_tx = base64.b64encode(signed_bytes).decode()
                async with aiohttp.ClientSession() as jito_session:
                    async with jito_session.post(
                        JITO_BUNDLE_URL,
                        json={"transactions": [signed_tx]},
                        headers={"Authorization": f"Bearer {jito_key}"},
                        timeout=10,
                    ) as bundle_resp:
                        bundle_resp.raise_for_status()
                        bundle_data = await bundle_resp.json()
                bundle_id = bundle_data["bundleId"]
                start = asyncio.get_event_loop().time()
                poll_timeout = config.get("jito_poll_timeout", 15)
                while True:
                    try:
                        status_data = await api_helpers.fetch_jito_bundle(bundle_id, jito_key)
                    except Exception as err:
                        logger.warning("Jito bundle fetch failed: %s", err)
                        break
                    txs = (
                        status_data.get("transactions")
                        or status_data.get("bundle", {}).get("transactions")
                        or []
                    )
                    sigs = [t.get("signature") for t in txs if t.get("signature")]
                    landed = (
                        status_data.get("landed")
                        or status_data.get("status") == "Landed"
                        or status_data.get("bundle", {}).get("state") == "Landed"
                    )
                    if landed and sigs:
                        tx_hash = sigs[0]
                        break
                    if asyncio.get_event_loop().time() - start > poll_timeout:
                        logger.warning("Jito bundle %s did not land in time", bundle_id)
                        break
                    await asyncio.sleep(1)
            except Exception as err:
                logger.warning("Jito submission failed: %s", err)
        if tx_hash is None:
            logger.info("Falling back to send_raw_transaction")
            for attempt in range(max_retries):
                try:
                    if signed_bytes and hasattr(client, "send_raw_transaction"):
                        send_res = await client.send_raw_transaction(signed_bytes)
                    else:
                        from solana.rpc.api import Client as SyncClient
                        send_res = SyncClient(rpc_url).send_transaction(tx, keypair)
                    tx_hash = send_res["result"]
                    break
                except Exception as err:
                    if "congestion" in str(err).lower() and attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    raise
            if tx_hash is None:
                raise RuntimeError("Swap failed after retries")

        poll_timeout = config.get("poll_timeout", 60)
        confirm_res = None
        try:
            confirm_res = await asyncio.wait_for(
                client.confirm_transaction(tx_hash, commitment="confirmed"),
                timeout=poll_timeout,
            )
        except Exception:
            for attempt in range(3):
                try:
                    async with AsyncClient(rpc_url) as aclient:
                        confirm_res = await asyncio.wait_for(
                            aclient.confirm_transaction(tx_hash, commitment="confirmed"),
                            timeout=poll_timeout,
                        )
                    break
                except Exception as err:
                    if attempt < 2:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    err_msg = notifier.notify(f"Confirmation failed for {tx_hash}")
                    if err_msg:
                        logger.error("Failed to send message: %s", err_msg)
                    raise TimeoutError("Transaction confirmation failed") from err
        for attempt in range(3):
            try:
                confirm_res = await asyncio.wait_for(
                    client.confirm_transaction(tx_hash, commitment="confirmed"),
                    timeout=poll_timeout,
                )
                break
            except Exception as err:
                if attempt < 2:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                err_msg = notifier.notify(f"Confirmation failed for {tx_hash}")
                if err_msg:
                    logger.error("Failed to send message: %s", err_msg)
                raise TimeoutError("Transaction confirmation failed") from err

    status = None
    if isinstance(confirm_res, dict):
        status = confirm_res.get("status") or confirm_res.get("value", {}).get("confirmationStatus")
    result = {
        "token_in": token_in,
        "token_out": token_out,
        "amount": amount,
        "tx_hash": tx_hash,
        "route": route,
        "status": status or "confirmed",
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
