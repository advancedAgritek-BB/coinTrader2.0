from __future__ import annotations

"""Utility to score newly issued tokens for early pump potential.

The :func:`assess_early_token` coroutine aggregates lightweight on‑chain
metrics, social sentiment, basic volume checks and an optional machine
learning model from the optional ``coinTrader_Trainer`` package.  The
resulting score ranges from ``0.0`` to ``1.0`` where higher values
indicate a more attractive candidate for early trading.
"""

from typing import Any, Dict
import asyncio

import pandas as pd
import requests

from crypto_bot.sentiment_filter import fetch_twitter_sentiment_async
from crypto_bot.utils.market_loader import fetch_geckoterminal_ohlcv
from crypto_bot.utils.logger import LOG_DIR, setup_logger

try:  # optional dependency
    from coinTrader_Trainer.ml_trainer import load_model, predict_regime  # type: ignore
except Exception:  # pragma: no cover - optional package missing
    load_model = None  # type: ignore
    predict_regime = None  # type: ignore

logger = setup_logger(__name__, LOG_DIR / "early_assessment.log")


async def assess_early_token(symbol: str, mint: str, cfg: Dict[str, Any]) -> float:
    """Return an early pump score for ``symbol``.

    The score combines:

    * **On-chain checks** using Helius metadata (developer holdings,
      initial liquidity).
    * **Sentiment** from Twitter via :func:`fetch_twitter_sentiment_async`.
    * **Recent volume** from GeckoTerminal OHLCV data.
    * **Machine learning** prediction from ``coinTrader_Trainer`` when
      available.

    The final score ranges from ``0`` to ``1``.  Scores above ``0.6``
    are considered promising candidates for further evaluation.
    """

    score = 0.0

    # --- On-chain filters -------------------------------------------------
    helius_api = cfg.get("helius_api_key")
    if helius_api:
        url = f"https://api.helius.xyz/v0/token-metadata?api-key={helius_api}&mints={mint}"
        try:
            resp = await asyncio.to_thread(requests.get, url, timeout=10)
            data = resp.json().get(mint, {}) if resp.ok else {}
        except Exception as exc:  # pragma: no cover - network best effort
            logger.error("Helius token lookup failed for %s: %s", mint, exc)
            data = {}

        try:
            dev_hold_pct = (
                float(data.get("dev_holding", 0))
                / float(data.get("total_supply", 1))
                * 100.0
            )
        except Exception:
            dev_hold_pct = 0.0

        if dev_hold_pct > 20:  # Rug risk
            return 0.0

        try:
            liquidity_usd = float(data.get("initial_liquidity_usd", 0) or 0.0)
        except Exception:
            liquidity_usd = 0.0

        if liquidity_usd < 5_000:
            return 0.0
        score += 0.3 if liquidity_usd > 10_000 else 0.1

    # --- Sentiment -------------------------------------------------------
    try:
        sentiment = await fetch_twitter_sentiment_async(query=symbol)
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("Sentiment fetch failed for %s: %s", symbol, exc)
        sentiment = 50
    logger.info("Sentiment for %s: %d", symbol, sentiment)
    score += (sentiment / 100.0) * 0.3

    # --- Volume via GeckoTerminal ---------------------------------------
    vol_usd = 0.0
    try:
        ohlcv = await fetch_geckoterminal_ohlcv(f"{symbol}/USDC", limit=5)
        if ohlcv:
            vol_usd = float(ohlcv[-1][5])
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("GeckoTerminal fetch failed for %s: %s", symbol, exc)
    if vol_usd > 10_000:
        score += 0.2

    # --- ML prediction ---------------------------------------------------
    if load_model and predict_regime:
        try:
            model = load_model("xrpusd_regime_lgbm")
            # Minimal feature set; additional features can be added as needed.
            early_df = pd.DataFrame(
                [{"volume": vol_usd, "sentiment": sentiment}]
            )
            regime_prob = predict_regime(early_df, model).get("volatile_pump", 0.0)
            score += float(regime_prob) * 0.4
        except Exception as exc:  # pragma: no cover - optional
            logger.error("ML regime prediction failed for %s: %s", symbol, exc)

    logger.info("Early score for %s: %.2f", symbol, score)
    return float(score)


__all__ = ["assess_early_token"]
