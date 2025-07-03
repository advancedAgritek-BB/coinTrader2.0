class CapitalTracker:
    """Track capital usage per strategy."""

    def __init__(self, allocation: dict | None = None) -> None:
        self.allocation = allocation or {}
        self._usage: dict[str, float] = {k: 0.0 for k in self.allocation}

    def can_allocate(self, strategy: str, amount: float, balance: float) -> bool:
        """Return True if ``amount`` can be allocated to ``strategy``."""
        cap = self.allocation.get(strategy)
        if cap is None:
            return True
        allowed = balance * cap
        return self._usage.get(strategy, 0.0) + amount <= allowed

    def allocate(self, strategy: str, amount: float) -> None:
        """Record capital allocation for ``strategy``."""
        self._usage[strategy] = self._usage.get(strategy, 0.0) + amount

    def deallocate(self, strategy: str, amount: float) -> None:
        """Remove allocation when a position is closed."""
        self._usage[strategy] = max(0.0, self._usage.get(strategy, 0.0) - amount)

    def update_allocation(self, allocation: dict) -> None:
        """Replace allocation limits with ``allocation`` and reset missing keys."""
        self.allocation = allocation
        for strat in allocation:
            self._usage.setdefault(strat, 0.0)
        for strat in list(self._usage):
            if strat not in allocation:
                self._usage.pop(strat)
