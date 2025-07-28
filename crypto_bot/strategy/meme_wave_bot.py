from __future__ import annotations

import os
from typing import Optional, Tuple

import aiohttp
import pandas as pd

from crypto_bot.utils.volatility import normalize_score_by_volatility
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor


WEB_SEARCH_URL = os.getenv("WEB_SEARCH_URL", "https://api.example.com/search")


async def web_search(query: str) -> int:
    """Return the number of results for ``query`` using a web API."""
    from urllib.parse import quote_plus

    url = f"{WEB_SEARCH_URL}?q={quote_plus(query)}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return 0
                data = await resp.json()
                count = data.get("count", 0)
                try:
                    return int(count)
                except Exception:
                    return 0
    except Exception:
        return 0


async def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    *,
    symbol: str | None = None,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
) -> Tuple[float, str]:
    """Return a meme-wave trading signal based on momentum and social buzz."""

    if df is None or df.empty:
        return 0.0, "none"

    params = config.get("meme_wave", {}) if config else {}
    lookback = int(params.get("lookback", 20))
    if len(df) < lookback:
        return 0.0, "none"

    recent = df.iloc[-(lookback + 1) :]
    ma = recent["close"].ewm(span=lookback, adjust=False).mean()
    momentum = recent["close"].iloc[-1] - ma.iloc[-1]

    direction = "none"
    if momentum > 0:
        direction = "long"
    elif momentum < 0:
        direction = "short"

    score = abs(momentum) / ma.iloc[-1] if ma.iloc[-1] else 0.0

    query = symbol or params.get("query")
    if query:
        mentions = await web_search(query)
        min_mentions = int(params.get("min_mentions", 0))
        if mentions < min_mentions:
            return 0.0, "none"
        weight = float(params.get("mention_weight", 0.01))
        score += mentions * weight

    if mempool_monitor is not None:
        try:
            fee = mempool_monitor.fetch_priority_fee()
            threshold = float(params.get("fee_threshold", 30))
            if fee >= threshold:
                return 0.0, "none"
        except Exception:
            pass

    score = min(score, 1.0)
    if score > 0:
        score = normalize_score_by_volatility(recent, score)

    return score, direction


class regime_filter:
    """Match trending or volatile regimes."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime in {"trending", "volatile"}
"""Solana meme wave strategy using simple volume surge detection."""

import asyncio
from typing import Optional, Tuple, Mapping

import pandas as pd
import ta

from crypto_bot.solana_trading import sniper_trade
from crypto_bot.solana.exit import monitor_price
from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor


async def trade(symbol: str, amount: float, cfg: Mapping[str, object]) -> dict:
    """Execute a meme-wave trade using :func:`sniper_trade`."""
    wallet = str(cfg.get("wallet_address", ""))
    base, quote = symbol.split("/")
    return await sniper_trade(
        wallet,
        quote,
        base,
        amount,
        dry_run=bool(cfg.get("dry_run", True)),
        slippage_bps=int(cfg.get("slippage_bps", 50)),
        notifier=cfg.get("notifier"),
        mempool_monitor=cfg.get("mempool_monitor"),
        mempool_cfg=cfg.get("mempool_cfg"),
    )


async def exit_trade(price_feed, entry_price: float, cfg: Mapping[str, float]) -> dict:
    """Exit a meme-wave trade using :func:`monitor_price`."""
    return await monitor_price(price_feed, entry_price, cfg)


def generate_signal(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    *,
    mempool_monitor: Optional[SolanaMempoolMonitor] = None,
    mempool_cfg: Optional[dict] = None,
) -> Tuple[float, str]:
    """Return score and direction based on volume and ATR expansion."""
    if df is None or df.empty:
        return 0.0, "none"

    params = config.get("meme_wave_bot", {}) if config else {}
    atr_window = int(params.get("atr_window", 14))
    vol_window = int(params.get("volume_window", 20))
    jump_mult = float(params.get("jump_mult", 3.0))
    vol_mult = float(params.get("volume_mult", 3.0))

    lookback = max(atr_window, vol_window)
    if len(df) < lookback + 1:
        return 0.0, "none"

    recent = df.tail(lookback + 1)
    atr = ta.volatility.average_true_range(
        recent["high"], recent["low"], recent["close"], window=atr_window
    )
    if atr.empty or pd.isna(atr.iloc[-1]):
        return 0.0, "none"

    price_change = recent["close"].iloc[-1] - recent["close"].iloc[-2]
    vol = recent["volume"].iloc[-1]
    avg_vol = recent["volume"].iloc[:-1].mean()

    if (
        abs(price_change) >= atr.iloc[-1] * jump_mult
        and avg_vol > 0
        and vol >= avg_vol * vol_mult
    ):
        return 1.0, "long" if price_change > 0 else "short"
    return 0.0, "none"


class regime_filter:
    """Match volatile regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "volatile"
