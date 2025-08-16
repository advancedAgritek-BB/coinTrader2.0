import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable


class Bandit:
    """Simple Thompson sampling bandit for strategy selection."""

    def __init__(
        self,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        explore_pct: float = 0.05,
        state_file: str = "crypto_bot/state/bandit.json",
    ) -> None:
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.explore_pct = float(explore_pct)
        self.state_file = Path(state_file)

        self.priors: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.update_count = 0
        self._load_state()

    def _load_state(self) -> None:
        """Load priors from ``state_file`` if present."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.priors = data.get("priors", {})
                self.update_count = int(data.get("count", 0))
            except Exception:
                self.priors = {}
                self.update_count = 0

    def _save_state(self) -> None:
        """Persist priors to ``state_file``."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"priors": self.priors, "count": self.update_count}
        self.state_file.write_text(json.dumps(data))

    def update(self, symbol: str, strategy: str, win: bool) -> None:
        """Update Beta priors for ``symbol`` and ``strategy``."""
        sym_data = self.priors.setdefault(symbol, {})
        stats = sym_data.setdefault(
            strategy,
            {"alpha": float(self.alpha0), "beta": float(self.beta0), "trades": 0},
        )
        if win:
            stats["alpha"] += 1
        else:
            stats["beta"] += 1
        stats["trades"] = stats.get("trades", 0) + 1
        self.update_count += 1
        # Persist state on every update so historical performance carries
        # across application restarts.
        self._save_state()

    def select(self, context: Any, arms: Iterable[str], symbol: str) -> str:
        """Return a strategy name chosen via Thompson sampling."""
        arms = list(arms)
        if not arms:
            raise ValueError("No arms to choose from")

        if random.random() < self.explore_pct:
            return random.choice(arms)

        sym_data = self.priors.setdefault(symbol, {})
        scores: Dict[str, float] = {}
        for arm in arms:
            stats = sym_data.setdefault(
                arm,
                {"alpha": float(self.alpha0), "beta": float(self.beta0), "trades": 0},
            )
            scores[arm] = random.betavariate(stats["alpha"], stats["beta"])
        return max(scores.items(), key=lambda x: x[1])[0]


bandit = Bandit()

__all__ = ["Bandit", "bandit"]
