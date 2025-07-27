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
    df: pd.DataFrame, config: Optional[dict] = None
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
