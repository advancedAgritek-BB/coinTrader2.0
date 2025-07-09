from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from crypto_bot.utils.logger import LOG_DIR


STATS_FILE = LOG_DIR / "strategy_stats.json"
WEIGHTS_FILE = LOG_DIR / "weights.json"


class OnlineWeightOptimizer:
    """Simple online optimizer updating strategy weights from PnL."""

    def __init__(self, learning_rate: float = 0.1, min_weight: float = 0.0) -> None:
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.weights: Dict[str, float] = {}
        self._load_weights()

    def _load_stats(self) -> Dict[str, Dict]:
        if STATS_FILE.exists():
            try:
                return json.loads(STATS_FILE.read_text())
            except Exception:
                return {}
        return {}

    def _load_weights(self) -> None:
        if WEIGHTS_FILE.exists():
            try:
                self.weights = json.loads(WEIGHTS_FILE.read_text())
            except Exception:
                self.weights = {}

    def _save_weights(self) -> None:
        WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        WEIGHTS_FILE.write_text(json.dumps(self.weights))

    def update(self) -> None:
        """Update weights using the latest PnL data."""
        stats = self._load_stats()
        for name, record in stats.items():
            pnl = float(record.get("pnl", 0.0))
            w = self.weights.get(name, 1.0)
            w += self.learning_rate * pnl
            if w < self.min_weight:
                w = self.min_weight
            self.weights[name] = w
        self._normalize()
        self._save_weights()

    def _normalize(self) -> None:
        total = sum(self.weights.values())
        if not total:
            return
        for k in self.weights:
            self.weights[k] /= total

    def get_weights(self) -> Dict[str, float]:
        """Return the current normalized weights."""
        return self.weights


_optimizer = OnlineWeightOptimizer()


def get_optimizer(config: dict | None = None) -> OnlineWeightOptimizer:
    """Return global optimizer configured from ``config``."""
    if config is None:
        return _optimizer
    cfg = config.get("signal_weight_optimizer", {})
    _optimizer.learning_rate = cfg.get("learning_rate", 0.1)
    _optimizer.min_weight = cfg.get("min_weight", 0.0)
    if cfg.get("enabled"):
        _optimizer.update()
    return _optimizer

