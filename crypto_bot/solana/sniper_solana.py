from __future__ import annotations

from typing import Mapping, Optional, Tuple

import pandas as pd
import ta

from crypto_bot.utils.pyth_utils import get_pyth_price

from .risk import RiskTracker
from .safety import is_safe
from .score import score_event
from .watcher import NewPoolEvent


class RugCheckAPI:
    """Placeholder API returning a rug risk score between 0 and 1."""

    @staticmethod
    def risk_score(token: str) -> float:  # pragma: no cover - placeholder
        return 0.0


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    **_,
) -> Tuple[float, str]:
    """Return a signal score and direction based on ATR jumps."""

    if df is None or df.empty:
        return 0.0, "none"

    params = config or {}
    atr_window = int(params.get("atr_window", 14))
    jump_mult = float(params.get("jump_mult", 4.0))
    rug_threshold = float(params.get("rug_threshold", 0.5))
    profit_target = float(params.get("profit_target_pct", 0.05))
    token = params.get("token", "")
    entry_price = params.get("entry_price")
    is_trading = bool(params.get("is_trading", True))
    conf_pct = float(params.get("conf_pct", 0.0))

    if not is_trading or conf_pct > 0.5:
        return 0.0, "none"

    if len(df) < atr_window + 1:
        return 0.0, "none"

    if token:
        price = get_pyth_price(f"Crypto.{token}/USD", config)
        try:
            df = df.copy()
            df.at[df.index[-1], "close"] = float(price)
        except Exception:  # pragma: no cover - defensive
            pass

    atr = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=atr_window
    )
    if atr.empty or pd.isna(atr.iloc[-1]):
        return 0.0, "none"

    price_change = df["close"].iloc[-1] - df["close"].iloc[-2]
    if abs(price_change) >= atr.iloc[-1] * jump_mult:
        direction = "long" if price_change > 0 else "short"
        if token and RugCheckAPI.risk_score(token) >= rug_threshold:
            return 0.0, "none"
        return 1.0, direction

    if entry_price is not None:
        if df["close"].iloc[-1] >= float(entry_price) * (1 + profit_target):
            return 1.0, "close"

    return 0.0, "none"


def score_new_pool(
    event: NewPoolEvent,
    config: Mapping[str, object],
    risk_tracker: RiskTracker,
) -> Tuple[float, str]:
    """Return a score and direction for a new pool event.

    Parameters
    ----------
    event:
        Pool creation event to evaluate.
    config:
        Configuration mapping with ``scoring``, ``safety`` and ``risk``
        subsections. Optionally includes ``twitter_score``.
    risk_tracker:
        Tracker for enforcing risk limits.
    """

    if not is_safe(event, config.get("safety", {})):
        return 0.0, "none"

    if not risk_tracker.allow_snipe(event.token_mint, config.get("risk", {})):
        return 0.0, "none"

    scoring_cfg = config.get("scoring", {})
    score = score_event(event, scoring_cfg)

    sentiment = float(config.get("twitter_score", 0))
    weight = float(scoring_cfg.get("twitter_weight", 1.0))
    score += sentiment * weight

    return score, "long"


class regime_filter:
    """Match volatile regime on Solana."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "volatile"


import asyncio
import json
from typing import Dict, Any, Optional
import aiohttp
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from solders.rpc.responses import (
    GetTokenAccountBalanceResp,
    GetAccountInfoResp,
)
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__, "crypto_bot/logs/solana_rug_check.log")

# Constants
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
RUGCHECK_API_URL = "https://api.rugcheck.xyz/v1/tokens/{token}/report"
DEXSCREENER_API_URL = "https://api.dexscreener.com/latest/dex/tokens/{token}"
RAYDIUM_PROGRAM_ID = Pubkey.from_string(
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Nyh"
)

# Risk thresholds
MIN_LIQUIDITY_USD = 10000
MAX_TOP_HOLDER_PCT = 20.0
MAX_TOP_10_HOLDERS_PCT = 50.0
MIN_TOKEN_AGE_MINUTES = 5
RUGCHECK_RISK_THRESHOLD = 50


async def fetch_token_metadata(
    client: AsyncClient, token_mint: str
) -> Optional[Dict[str, Any]]:
    """Fetch token metadata like mint authority and supply."""
    try:
        mint_pubkey = Pubkey.from_string(token_mint)
        resp: GetAccountInfoResp = await client.get_account_info(mint_pubkey)
        if resp.value is None:
            return None

        account_data = resp.value.data
        mint_authority = (
            account_data[0:32] if account_data[32:36] == b"\x01" else None
        )
        freeze_authority = (
            account_data[36:68] if account_data[68:72] == b"\x01" else None
        )
        supply = int.from_bytes(account_data[4:12], "little")

        return {
            "mint_authority": bool(mint_authority),
            "freeze_authority": bool(freeze_authority),
            "total_supply": supply,
        }
    except Exception as e:  # pragma: no cover - network errors
        logger.error(f"Error fetching token metadata for {token_mint}: {e}")
        return None


async def fetch_dexscreener_data(
    session: aiohttp.ClientSession, token_mint: str
) -> Optional[Dict[str, Any]]:
    """Fetch liquidity and holder info from DexScreener."""
    url = DEXSCREENER_API_URL.format(token=token_mint)
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            pairs = data.get("pairs", [])
            if not pairs:
                return None

            main_pair = pairs[0]
            liquidity_usd = main_pair.get("liquidity", {}).get("usd", 0)
            top_10_holders_pct = main_pair.get("top10HoldersPercent", 0)
            fdv = main_pair.get("fdv", 0)

            return {
                "liquidity_usd": liquidity_usd,
                "top_10_holders_pct": top_10_holders_pct,
                "fdv": fdv,
            }
    except Exception as e:  # pragma: no cover - network errors
        logger.error(f"Error fetching DexScreener data for {token_mint}: {e}")
        return None


async def fetch_rugcheck_score(
    session: aiohttp.ClientSession, token_mint: str
) -> Optional[int]:
    """Fetch risk score from RugCheck (0-100, higher risk)."""
    url = RUGCHECK_API_URL.format(token=token_mint)
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return data.get("riskScore")
    except Exception as e:  # pragma: no cover - network errors
        logger.error(f"Error fetching RugCheck score for {token_mint}: {e}")
        return None


async def check_liquidity_locked(client: AsyncClient, pool_address: str) -> bool:
    """Return ``True`` if pool tokens are burned or locked."""
    try:
        pool_pubkey = Pubkey.from_string(pool_address)
        resp: GetTokenAccountBalanceResp = await client.get_token_account_balance(
            pool_pubkey
        )
        if resp.value is None:
            return False
        balance = resp.value.ui_amount
        return balance == 0
    except Exception as e:  # pragma: no cover - network errors
        logger.error(f"Error checking liquidity lock for pool {pool_address}: {e}")
        return False


async def simulate_honeypot(client: AsyncClient, token_mint: str) -> bool:
    """Basic honeypot detection via mint authority."""
    metadata = await fetch_token_metadata(client, token_mint)
    if metadata and metadata.get("mint_authority"):
        return True
    return False


async def rug_check(
    token_mint: str, pool_address: Optional[str] = None
) -> Dict[str, Any]:
    """Comprehensive rug check returning a risk score."""

    async with AsyncClient(SOLANA_RPC_URL) as client:
        async with aiohttp.ClientSession() as session:
            tasks = [
                fetch_token_metadata(client, token_mint),
                fetch_dexscreener_data(session, token_mint),
                fetch_rugcheck_score(session, token_mint),
                simulate_honeypot(client, token_mint),
            ]
            if pool_address:
                tasks.append(check_liquidity_locked(client, pool_address))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            metadata, dexscreener, rug_score, is_honeypot = results[:4]
            lp_locked = results[4] if pool_address else False

            details: Dict[str, Any] = {}
            risk_factors = 0
            total_checks = 5

            if not isinstance(metadata, dict):
                details["metadata_error"] = str(metadata)
                risk_factors += 1
            else:
                if metadata["mint_authority"]:
                    details["mint_authority"] = "Exists (risk: can mint more)"
                    risk_factors += 1
                if metadata["freeze_authority"]:
                    details["freeze_authority"] = "Exists (risk: can freeze accounts)"
                    risk_factors += 1

            if not isinstance(dexscreener, dict):
                details["dexscreener_error"] = str(dexscreener)
                risk_factors += 1
            else:
                liquidity = dexscreener.get("liquidity_usd", 0)
                details["liquidity_usd"] = liquidity
                if liquidity < MIN_LIQUIDITY_USD:
                    details["low_liquidity"] = f"{liquidity} < {MIN_LIQUIDITY_USD}"
                    risk_factors += 1

                top_10_pct = dexscreener.get("top_10_holders_pct", 0)
                details["top_10_holders_pct"] = top_10_pct
                if top_10_pct > MAX_TOP_10_HOLDERS_PCT:
                    details["concentrated_holders"] = (
                        f"{top_10_pct}% > {MAX_TOP_10_HOLDERS_PCT}%"
                    )
                    risk_factors += 1

            if not isinstance(rug_score, int):
                details["rugcheck_error"] = str(rug_score)
                risk_factors += 1
            else:
                details["rugcheck_score"] = rug_score
                if rug_score > RUGCHECK_RISK_THRESHOLD:
                    details["high_rug_score"] = f"{rug_score} > {RUGCHECK_RISK_THRESHOLD}"
                    risk_factors += 1

            if isinstance(is_honeypot, Exception):
                details["honeypot_error"] = str(is_honeypot)
                risk_factors += 1
            elif is_honeypot:
                details["potential_honeypot"] = True
                risk_factors += 1

            if pool_address:
                details["lp_locked"] = lp_locked
                if not lp_locked:
                    details["lp_not_locked"] = "Liquidity not burned/locked"
                    risk_factors += 1
                total_checks += 1

            risk_score = risk_factors / total_checks
            safe = risk_score < 0.3

            return {
                "safe": safe,
                "risk_score": risk_score,
                "details": details,
            }


async def _get_price_volume(mint: str) -> tuple[float, float]:
    """Return current price and 24h volume for ``mint`` using DexScreener."""

    url = DEXSCREENER_API_URL.format(token=mint)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
        pair = (data.get("pairs") or [{}])[0]
        price = float(pair.get("priceUsd") or 0.0)
        volume = float((pair.get("volume") or {}).get("h24") or 0.0)
    except Exception:  # pragma: no cover - network errors
        price, volume = 0.0, 0.0
    return price, volume


async def execute_snipe(
    mint: str, cfg: Mapping[str, Any], exchange: object
) -> Dict[str, Any]:
    """Buy a token and exit on trailing stop or volume dump."""

    from crypto_bot.execution.cex_executor import (
        execute_trade_async as cex_trade_async,
    )
    from crypto_bot.fund_manager import auto_convert_funds

    symbol = cfg.get("symbol", "")
    market = symbol if "/" in symbol else f"{symbol}/USDC"
    wallet_balance = float(cfg.get("wallet_balance", 0))
    trade_size = float(cfg.get("trade_size_pct", 0)) * wallet_balance
    if trade_size <= 0:
        return {"error": "invalid_trade_size"}

    await cex_trade_async(exchange, market, "buy", trade_size)

    price, volume = await _get_price_volume(mint)
    peak_price = price
    peak_volume = volume
    trailing_pct = float(cfg.get("trailing_stop_pct", 10))
    volume_drop_pct = float(cfg.get("volume_drop_pct", 50))
    poll = float(cfg.get("poll_interval", 1))

    while True:
        await asyncio.sleep(poll)
        price, volume = await _get_price_volume(mint)
        if price <= 0:
            continue
        if price > peak_price:
            peak_price = price
        if volume > peak_volume:
            peak_volume = volume

        price_drop = (peak_price - price) / peak_price if peak_price else 0.0
        vol_drop = (peak_volume - volume) / peak_volume if peak_volume else 0.0
        if trailing_pct and price_drop >= trailing_pct / 100:
            await cex_trade_async(exchange, market, "sell", trade_size)
            await auto_convert_funds("USDC", "BTC", exchange)
            return {"exit": "trailing", "price": price}
        if volume_drop_pct and vol_drop >= volume_drop_pct / 100:
            await cex_trade_async(exchange, market, "sell", trade_size)
            await auto_convert_funds("USDC", "BTC", exchange)
            return {"exit": "volume", "price": price}
