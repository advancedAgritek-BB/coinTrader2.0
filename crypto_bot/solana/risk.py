"""Risk tracking for Solana snipes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping


@dataclass
class RiskState:
    """Persistent risk state."""

    daily_loss: float = 0.0
    active_snipes: Dict[str, float] = field(default_factory=dict)


class RiskTracker:
    """Utility for enforcing risk limits across snipes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.state = RiskState()
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open() as fh:
                    data = json.load(fh)
                self.state = RiskState(**data)
            except Exception:
                self.state = RiskState()

    def save(self) -> None:
        try:
            with self.path.open("w") as fh:
                json.dump(self.state.__dict__, fh)
        except Exception:
            pass

    def allow_snipe(self, mint: str, risk_cfg: Mapping[str, float]) -> bool:
        """Return ``True`` if a new snipe is allowed under ``risk_cfg``."""

        max_concurrent = int(risk_cfg.get("max_concurrent", 1))
        if len(self.state.active_snipes) >= max_concurrent:
            return False

        loss_cap = float(risk_cfg.get("daily_loss_cap", 0))
        if loss_cap and self.state.daily_loss >= loss_cap:
            return False

        return True

    def record_loss(self, mint: str, amount: float) -> None:
        self.state.daily_loss += amount
        self.state.active_snipes.pop(mint, None)
        self.save()

    def add_snipe(self, mint: str, amount: float) -> None:
        self.state.active_snipes[mint] = amount
        self.save()
