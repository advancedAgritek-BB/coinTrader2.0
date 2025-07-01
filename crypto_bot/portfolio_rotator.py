"""Utility for rotating portfolio holdings based on momentum or Sharpe scores."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yaml

from crypto_bot.fund_manager import auto_convert_funds
from crypto_bot.utils.logger import setup_logger


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
LOG_FILE = Path("crypto_bot/logs/rotations.json")
SCORE_FILE = Path("crypto_bot/logs/asset_scores.json")


class PortfolioRotator:
    """Score assets and rebalance holdings toward the top performers."""

    def __init__(self) -> None:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        self.config = cfg.get("portfolio_rotation", {})
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(__name__, "crypto_bot/logs/portfolio_rotation.log")

    def score_assets(
        self,
        exchange,
        symbols: Iterable[str],
        lookback_days: int,
        method: str,
    ) -> Dict[str, float]:
        """Return a score for each symbol using Sharpe ratio or momentum."""

        scores: Dict[str, float] = {}
        for sym in symbols:
            try:
                ohlcv = exchange.fetch_ohlcv(sym, timeframe="1d", limit=lookback_days)
            except Exception as exc:  # pragma: no cover - network
                self.logger.error("OHLCV fetch failed for %s: %s", sym, exc)
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            if method == "sharpe":
                rets = df["close"].pct_change().dropna()
                std = rets.std()
                score = float(rets.mean() / std) if std else 0.0
            else:  # momentum
                score = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
            scores[sym] = score
            self.logger.info("Score for %s: %.4f", sym, score)

        return scores

    async def rotate(
        self,
        exchange,
        wallet: str,
        current_holdings: Dict[str, float],
        telegram_token: str = "",
        chat_id: str = "",
    ) -> Dict[str, float]:
        """Rebalance holdings toward the highest scored assets."""

        method = self.config.get("scoring_method", "sharpe")
        lookback = self.config.get("lookback_days", 30)
        threshold = self.config.get("rebalance_threshold", 0.0)
        top_n = self.config.get("top_assets", len(current_holdings))

        scores = self.score_assets(exchange, current_holdings.keys(), lookback, method)
        self._log_scores(scores)
        if not scores:
            return current_holdings

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        desired = [s for s, _ in ranked[:top_n]]

        new_alloc = current_holdings.copy()
        for token, amount in list(current_holdings.items()):
            if token in desired or amount <= 0:
                continue
            # choose best asset not currently held
            candidates = [d for d in desired if d not in current_holdings]
            if not candidates:
                break
            target = candidates[0]
            improvement = scores.get(target, 0) - scores.get(token, 0)
            if improvement <= threshold:
                continue

            self.logger.info(
                "Rotating %s -> %s amount %.4f (score diff %.4f)",
                token,
                target,
                amount,
                improvement,
            )
            # execute swap via fund manager helper
            await auto_convert_funds(
                wallet,
                token,
                target,
                amount,
                dry_run=True,
                telegram_token=telegram_token,
                chat_id=chat_id,
            )
            new_alloc.pop(token)
            new_alloc[target] = new_alloc.get(target, 0) + amount

        self._log_allocation(new_alloc)
        return new_alloc

    def _log_allocation(self, allocation: Dict[str, float]) -> None:
        """Append allocation to the rotation log."""
        data: List[Dict[str, float]]
        if LOG_FILE.exists():
            data = json.loads(LOG_FILE.read_text())
        else:
            data = []
        data.append(allocation)
        LOG_FILE.write_text(json.dumps(data))

    def _log_scores(self, scores: Dict[str, float]) -> None:
        """Write the latest asset scores to file."""
        SCORE_FILE.write_text(json.dumps(scores))

