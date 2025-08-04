from __future__ import annotations

import asyncio
from typing import Mapping, Optional, Tuple

import pandas as pd
import ta

from crypto_bot.execution.solana_mempool import SolanaMempoolMonitor
from crypto_bot.sentiment_filter import fetch_twitter_sentiment
from crypto_bot.solana.exit import monitor_price
from crypto_bot.solana_trading import sniper_trade
from crypto_bot.utils.volatility import normalize_score_by_volatility


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
    """Return a meme wave score and direction using volume and sentiment."""

    if mempool_monitor is None:
        return 0.0, "none"

    params = config.get("meme_wave_bot", {}) if config else {}
    vol_threshold = float(params.get("volume_threshold", 1.0))
    sentiment_thr = float(params.get("sentiment_threshold", 0.0))
    query = params.get("twitter_query") or ""
    atr_window = int(params.get("atr_window", 14))
    vol_window = int(params.get("volume_window", 20))
    jump_mult = float(params.get("jump_mult", 3.0))
    vol_mult = float(params.get("volume_mult", 3.0))
    vol_spike_thr = params.get("vol_spike_thr")

    recent_vol = mempool_monitor.get_recent_volume()
    avg_vol = mempool_monitor.get_average_volume()

    try:
        import asyncio
        import inspect

        if inspect.iscoroutine(recent_vol):
            recent_vol = asyncio.run(recent_vol)
        if inspect.iscoroutine(avg_vol):
            avg_vol = asyncio.run(avg_vol)
    except Exception:
        pass

    sentiment = fetch_twitter_sentiment(query) / 100.0

    if avg_vol and recent_vol >= avg_vol * vol_threshold and sentiment >= sentiment_thr:
        return 1.0, "long"

    spike = (
        abs(price_change) >= atr.iloc[-1] * jump_mult
        and avg_vol > 0
        and vol >= avg_vol * vol_mult
    )

    if not spike:
        return 0.0, "none"

    mempool_ok = True
    if mempool_monitor is not None and vol_spike_thr is not None:
        try:
            import asyncio

            try:
                recent_vol = asyncio.run(mempool_monitor.get_recent_volume())
                avg_mempool = asyncio.run(mempool_monitor.get_average_volume())
            except RuntimeError:
                recent_vol = 0.0
                avg_mempool = 0.0
        except Exception:
            recent_vol = 0.0
            avg_mempool = 0.0

        if avg_mempool <= 0 or recent_vol < float(vol_spike_thr) * avg_mempool:
            mempool_ok = False

    sentiment_ok = True
    if sentiment_thr is not None:
        try:
            from crypto_bot.sentiment_filter import fetch_twitter_sentiment

            q = query
            if not q:
                q = config.get("symbol") if isinstance(config, dict) else None
            sentiment = fetch_twitter_sentiment(q or "") / 100.0
            if sentiment < float(sentiment_thr):
                sentiment_ok = False
        except Exception:
            sentiment_ok = False

    if mempool_ok and sentiment_ok:
        return 1.0, "long" if price_change > 0 else "short"

    return 0.0, "none"


class regime_filter:
    """Match volatile regime."""

    @staticmethod
    def matches(regime: str) -> bool:
        return regime == "volatile"
