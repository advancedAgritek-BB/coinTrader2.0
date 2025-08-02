from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, Optional

import aiohttp
import numpy as np
from solana.rpc.async_api import AsyncClient

from crypto_bot.execution.solana_executor import (
    execute_swap,
    JUPITER_QUOTE_URL,
)
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
from crypto_bot.execution.cex_executor import execute_trade_async as cex_trade_async
from crypto_bot.fund_manager import auto_convert_funds
from crypto_bot.utils.logger import LOG_DIR, setup_logger

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model
except Exception:  # pragma: no cover - fallback when trainer missing
    load_model = None  # type: ignore[misc]


logger = setup_logger(__name__, LOG_DIR / "solana_trading.log")


async def _fetch_price(token_in: str, token_out: str, max_retries: int = 3) -> float:
    """Return current price for ``token_in``/``token_out`` using Jupiter.

    Parameters
    ----------
    max_retries:
        Maximum attempts when the price request fails. Defaults to ``3``.
    """
    proxy_url = "https://helius-proxy.raydium.io/api/v1/"
    request_url = proxy_url + JUPITER_QUOTE_URL.lstrip("/")

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    request_url,
                    params={
                        "inputMint": token_in,
                        "outputMint": token_out,
                        "amount": 1_000_000,
                        "slippageBps": 50,
                    },
                    timeout=10,
                ) as resp:
                    if resp.status == 401:
                        logger.error("401 in price fetch: Invalid Helius key")
                        return 0.0
                    resp.raise_for_status()
                    data = await resp.json()
            break
        except Exception as e:
            logger.error(f"Price fetch error (attempt {attempt+1}): {e}")
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
    """Return profit when price increase and model predicts breakout regime.

    The function monitors the swap price for up to 5 minutes. Each new
    quote is fed into the ``regime_lgbm`` model from ``coinTrader_Trainer``;
    profit is only returned when the price change is above ``threshold`` and
    the model identifies a breakout regime.

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

    try:  # optional ML model
        from coinTrader_Trainer.ml_trainer import load_model as trainer_load_model

        ml_model = trainer_load_model("profit")
        try:
            ml_model.predict([[0.0]])
        except Exception:  # pragma: no cover - best effort
            pass
    except Exception:  # pragma: no cover - optional dependency
        ml_model = None
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

        if load_model is None:
            logger.error("ML model loader not available")
            return 0.0
        model = load_model("regime_lgbm")

        start = time.time()
        while time.time() - start < 300:
            price = await _fetch_price(out_mint, in_mint, max_retries=3)
            if price:
                change = (price - entry_price) / entry_price
                features = [change, out_amount]
                prediction = model.predict([features])[0]
                if change >= threshold and np.argmax(prediction) == 2:
                    logger.info("Profit threshold hit with breakout regime: %s", change)
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
    config: dict | None = None,
) -> Dict:
    """Buy ``target_token`` then convert profits when threshold reached."""

    if mempool_monitor is None and (mempool_cfg or {}).get("enabled"):
        mempool_monitor = SolanaMempoolMonitor()

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

    if config:
        from crypto_bot.solana.exit import quick_exit

        async def price_feed():
            return await _fetch_price(target_token, base_token, max_retries=3)

        entry_price = await price_feed()
        await quick_exit(price_feed, entry_price, config)

    profit = await monitor_profit(tx_sig, profit_threshold)
    if profit > 0:
        logger.info("Auto-converting %s %s profit to %s", profit, target_token, base_token)
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

    if mempool_monitor is None and (mempool_cfg or {}).get("enabled"):
        mempool_monitor = SolanaMempoolMonitor()

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
