"""Triangular or multi-hop arbitrage strategy.

This bot scans three-leg loops such as BTC/ETH -> ETH/USDT -> BTC/USDT and
looks for price inconsistencies.  If the implied conversion rate after fees
exceeds a configurable threshold the strategy emits a high-confidence buy
signal.  When available, an optional machine learning model from the
``coinTrader_Trainer`` package is used to refine the score.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, Mapping, Optional, Tuple

import ccxt
import pandas as pd

from crypto_bot.utils.market_loader import fetch_order_book_async
from crypto_bot.indicators.atr import calc_atr
from crypto_bot.utils.ml_utils import warn_ml_unavailable_once

try:  # pragma: no cover - optional dependency
    from coinTrader_Trainer.ml_trainer import load_model

    ML_AVAILABLE = True
except Exception:  # pragma: no cover - trainer missing
    ML_AVAILABLE = False
    load_model = None  # type: ignore
    warn_ml_unavailable_once()

MODEL = load_model("triangular_arb_bot") if ML_AVAILABLE else None


def _calc_rate(books: Iterable[dict]) -> float:
    """Return implied conversion rate for a triangular loop."""

    try:
        bid1 = float(books[0]["bids"][0][0])
        bid2 = float(books[1]["bids"][0][0])
        ask3 = float(books[2]["asks"][0][0])
    except Exception:
        return 0.0
    if bid1 <= 0 or bid2 <= 0 or ask3 <= 0:
        return 0.0
    return (bid1 * bid2) / ask3


def generate_signal(
    df: pd.DataFrame,
    config: Optional[Mapping[str, object]] = None,
    *,
    exchange=None,
) -> Tuple[float, str, float]:
    """Return triangular arbitrage signal.

    The function fetches order books for configured loops, evaluates the
    implied rate after accounting for fees and emits a buy signal when the
    spread exceeds the configured threshold.
    """

    if df is None or df.empty:
        return 0.0, "none", 0.0

    params = config.get("triangular_arb_bot", {}) if config else {}
    loops = params.get(
        "arb_pairs", [("BTC/ETH", "ETH/USDT", "BTC/USDT")]
    )
    fee_rate = float(params.get("fee_rate", 0.001))
    threshold = float(params.get("spread_threshold", 0.005))

    ex = exchange
    if ex is None:
        name = str(params.get("exchange", "kraken")).lower()
        try:  # pragma: no cover - best effort
            ex = getattr(ccxt, name)()
        except Exception:
            return 0.0, "none", 0.0

    for loop_pairs in loops:
        if not loop_pairs:
            continue

        async def _fetch():
            return await asyncio.gather(
                *(fetch_order_book_async(ex, p) for p in loop_pairs)
            )

        try:
            books = asyncio.run(_fetch())
        except RuntimeError:  # pragma: no cover - event loop already running
            loop = asyncio.get_event_loop()
            books = loop.run_until_complete(_fetch())
        except Exception:
            continue

        if any(not b for b in books):
            continue

        rate = _calc_rate(books)
        if rate <= 0:
            continue
        effective = rate * (1 - fee_rate) ** 3
        if effective <= 1 + threshold:
            continue

        score = min(effective - 1, 1.0)
        if MODEL is not None:
            try:  # pragma: no cover - best effort
                ml_score = float(MODEL.predict(df))
                score = (score + ml_score) / 2
            except Exception:
                pass
        atr = calc_atr(df)
        return score, "long", atr

    return 0.0, "none", 0.0


class regime_filter:
    """Match sideways regime."""

    @staticmethod
    def matches(regime: str) -> bool:  # pragma: no cover - simple
        return regime == "sideways"

