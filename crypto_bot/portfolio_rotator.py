"""Utility for rotating portfolio holdings based on momentum or Sharpe scores."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yaml

from crypto_bot.fund_manager import auto_convert_funds
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.telegram import TelegramNotifier



CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
LOG_FILE = LOG_DIR / "rotations.json"
SCORE_FILE = LOG_DIR / "asset_scores.json"


class PortfolioRotator:
    """Score assets and rebalance holdings toward the top performers."""

    def __init__(self) -> None:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        self.config = cfg.get("portfolio_rotation", {})
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(__name__, LOG_DIR / "portfolio_rotation.log")

    async def score_assets(
        self,
        exchange,
        symbols: Iterable[str],
        lookback_days: int,
        method: str,
    ) -> Dict[str, float]:
        """Return a score for each symbol using Sharpe ratio or momentum."""

        scores: Dict[str, float] = {}
        fetch_fn = getattr(exchange, "fetch_ohlcv", None)
        for sym in symbols:
            try:
                if asyncio.iscoroutinefunction(fetch_fn):
                    ohlcv = await fetch_fn(sym, timeframe="1d", limit=lookback_days)
                else:
                    ohlcv = await asyncio.to_thread(
                        fetch_fn, sym, timeframe="1d", limit=lookback_days
                    )
            except Exception as exc:  # pragma: no cover - network
                self.logger.exception(
                    "OHLCV fetch failed for %s timeframe 1d lookback %s", sym, lookback_days
                )
                continue

            if (
                not ohlcv
                or not isinstance(ohlcv, (list, tuple))
                or any(
                    not isinstance(row, (list, tuple)) or len(row) != 6
                    for row in ohlcv
                )
            ):
                self.logger.error("Invalid OHLCV for %s: %r", sym, ohlcv)
                continue

            df = pd.DataFrame(
                ohlcv,
                columns=["ts", "open", "high", "low", "close", "volume"],
            )
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
        notifier: TelegramNotifier | None = None,
    ) -> Dict[str, float]:
        """Rebalance holdings toward the highest scored assets."""

        method = self.config.get("scoring_method", "sharpe")
        lookback = self.config.get("lookback_days", 30)
        threshold = self.config.get("rebalance_threshold", 0.0)
        top_n = self.config.get("top_assets", len(current_holdings))

        # convert holdings like {"BTC": 1} to trading pairs
        markets = getattr(exchange, "markets", {}) or {}
        quote_pref = self.config.get("quote_currency")
        quote_candidates = [quote_pref] if quote_pref else ["USD", "USDT"]
        pair_map: Dict[str, str] = {}
        for asset in current_holdings:
            if "/" in asset:
                pair_map[asset] = asset.split("/")[0]
                continue

            pair_found = None
            for quote in quote_candidates:
                if not quote:
                    continue
                pair = f"{asset}/{quote}"
                if pair in markets:
                    pair_found = pair
                    break
                if hasattr(exchange, "market"):
                    try:
                        if exchange.market(pair):
                            pair_found = pair
                            break
                    except Exception:  # pragma: no cover - best effort
                        pass
            if not pair_found:
                self.logger.warning("No matching pair for %s", asset)
                pair_map[asset] = asset
            else:
                pair_map[pair_found] = asset

        scores_pairs = await self.score_assets(exchange, pair_map.keys(), lookback, method)
        scores = {pair_map.get(p, p): s for p, s in scores_pairs.items()}
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
                notifier=notifier,
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

