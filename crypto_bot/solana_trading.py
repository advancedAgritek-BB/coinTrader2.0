import asyncio
import os
import time
from decimal import Decimal
from typing import Dict, Optional

import aiohttp
from solana.rpc.async_api import AsyncClient

from crypto_bot.execution.solana_executor import (
    execute_swap,
    JUPITER_QUOTE_URL,
)
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
from crypto_bot.execution.cex_executor import execute_trade_async as cex_trade_async
from crypto_bot.fund_manager import auto_convert_funds
from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "solana_trading.log")


_DECIMALS_CACHE: Dict[str, int] = {}


def to_base_units(amount: float | int, decimals: int) -> int:
    """Return ``amount`` converted to base units for ``decimals`` precision."""
    return int(Decimal(str(amount)) * (10 ** decimals))


async def get_decimals(mint: str) -> int:
    """Return cached decimals for ``mint`` using Solana RPC when needed."""
    cached = _DECIMALS_CACHE.get(mint)
    if cached is not None:
        return cached

    rpc_url = os.getenv(
        "SOLANA_RPC_URL",
        f"https://mainnet.helius-rpc.com/?api-key={os.getenv('HELIUS_KEY', '')}",
    )
    client = AsyncClient(rpc_url)
    try:
        resp = await client.get_token_supply(mint)
        decimals = resp.get("result", {}).get("value", {}).get("decimals", 0)
    except Exception:
        decimals = 0
    finally:
        await client.close()

    _DECIMALS_CACHE[mint] = decimals
    return decimals


async def _fetch_price(token_in: str, token_out: str, max_retries: int = 3) -> float:
    """Return current price for ``token_in``/``token_out`` using Jupiter.

    Parameters
    ----------
    max_retries:
        Maximum attempts when the price request fails. Defaults to ``3``.
    """
    decimals = await get_decimals(token_in)
    amount = to_base_units(1, decimals)
    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            try:
                async with session.get(
                    JUPITER_QUOTE_URL,
                    params={
                        "inputMint": token_in,
                        "outputMint": token_out,
                        "amount": amount,
                        "slippageBps": 50,
                    },
                    timeout=10,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                break
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return 0.0
    route = (data.get("data") or [{}])[0]
    try:
        return float(route["outAmount"]) / float(route["inAmount"])
    except Exception:
        return 0.0


async def monitor_profit(tx_sig: str, threshold: float = 0.2) -> float:
    """Return profit when price change exceeds ``threshold``.

    Parameters
    ----------
    tx_sig:
        Signature of the entry swap transaction.
    threshold:
        Percentage gain required to trigger profit taking.
    """

    rpc_url = os.getenv(
        "SOLANA_RPC_URL",
        f"https://mainnet.helius-rpc.com/?api-key={os.getenv('HELIUS_KEY', '')}",
    )
    client = AsyncClient(rpc_url)
    try:
        entry_price = None
        out_amount = 0.0
        in_mint = out_mint = ""
        # wait for confirmed tx with token balances
        for _ in range(30):
            try:
                resp = await client.get_confirmed_transaction(tx_sig, encoding="jsonParsed")
            except Exception:
                resp = None
            tx = resp.get("result") if resp else None
            if tx:
                meta = tx.get("meta", {})
                pre = meta.get("preTokenBalances") or []
                post = meta.get("postTokenBalances") or []
                if len(pre) >= 2 and len(post) >= 2:
                    in_mint = pre[0].get("mint", "")
                    out_mint = post[1].get("mint", "")
                    try:
                        in_amt = float(pre[0]["uiTokenAmount"].get("uiAmount", pre[0]["uiTokenAmount"].get("uiAmountString", 0)))
                        out_amount = float(post[1]["uiTokenAmount"].get("uiAmount", post[1]["uiTokenAmount"].get("uiAmountString", 0)))
                    except Exception:
                        in_amt = 0.0
                        out_amount = 0.0
                    if in_amt and out_amount:
                        entry_price = in_amt / out_amount
                        break
            await asyncio.sleep(2)
        if entry_price is None:
            return 0.0

        start = time.time()
        while time.time() - start < 300:
            price = await _fetch_price(out_mint, in_mint, max_retries=3)
            if price:
                change = (price - entry_price) / entry_price
                if change >= threshold:
                    return out_amount * change
            await asyncio.sleep(5)
    finally:
        await client.close()
    return 0.0


async def sniper_trade(
    wallet: str,
    base_token: str,
    target_token: str,
    amount: float,
    *,
    dry_run: bool = True,
    slippage_bps: int = 50,
    notifier: Optional[object] = None,
    profit_threshold: float = 0.2,
    mempool_monitor: SolanaMempoolMonitor | None = None,
    mempool_cfg: dict | None = None,
) -> Dict:
    """Buy ``target_token`` then convert profits when threshold reached."""

    trade = await execute_swap(
        base_token,
        target_token,
        amount,
        notifier=notifier,
        slippage_bps=slippage_bps,
        dry_run=dry_run,
        mempool_monitor=mempool_monitor,
        mempool_cfg=mempool_cfg,
    )
    tx_sig = trade.get("tx_hash")
    if not tx_sig or tx_sig == "DRYRUN":
        return trade

    profit = await monitor_profit(tx_sig, profit_threshold)
    if profit > 0:
        await auto_convert_funds(
            wallet,
            target_token,
            base_token,
            profit,
            dry_run=dry_run,
            slippage_bps=slippage_bps,
            notifier=notifier,
            mempool_monitor=mempool_monitor,
            mempool_cfg=mempool_cfg,
        )
    return trade


async def cross_chain_trade(
    exchange: object,
    ws_client: object | None,
    symbol: str,
    side: str,
    amount: float,
    *,
    dry_run: bool = True,
    slippage_bps: int = 50,
    use_websocket: bool = False,
    notifier: object | None = None,
    mempool_monitor: SolanaMempoolMonitor | None = None,
    mempool_cfg: dict | None = None,
    config: dict | None = None,
) -> Dict:
    """Execute a DEX trade then the opposite side on the CEX."""

    base, quote = symbol.split("/")
    from crypto_bot.utils.token_registry import TOKEN_MINTS

    base_mint = TOKEN_MINTS.get(base)
    quote_mint = TOKEN_MINTS.get(quote)
    if base_mint is None or quote_mint is None:
        logger.error("Unknown mint for %s", symbol)
        return {"error": "unknown_mint"}

    if side == "buy":
        price = await _fetch_price(quote_mint, base_mint, max_retries=3)
        in_amount = amount / price if price else amount
        token_in = quote_mint
        token_out = base_mint
        cex_side = "sell"
    else:
        in_amount = amount
        token_in = base_mint
        token_out = quote_mint
        cex_side = "buy"

    dex_trade = await execute_swap(
        token_in,
        token_out,
        in_amount,
        notifier=notifier,
        slippage_bps=slippage_bps,
        dry_run=dry_run,
        mempool_monitor=mempool_monitor,
        mempool_cfg=mempool_cfg,
        config=config,
    )

    cex_trade = await cex_trade_async(
        exchange,
        ws_client,
        symbol,
        cex_side,
        amount,
        notifier,
        dry_run=dry_run,
        use_websocket=use_websocket,
        config=config,
    )

    return {"dex": dex_trade, "cex": cex_trade}
