"""Solana meme wave strategy using simple volume surge detection."""
from __future__ import annotations

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
