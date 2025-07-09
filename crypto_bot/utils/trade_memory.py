"""Simple trade memory utilities for loss tracking."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Any

from .logger import LOG_DIR


LOG_FILE = LOG_DIR / "trade_memory.json"
MAX_LOSSES = 3
SLIPPAGE_THRESHOLD = 0.05
LOOKBACK_SECONDS = 3600


def configure(
    max_losses: int = 3,
    slippage_threshold: float = 0.05,
    lookback_seconds: int = 3600,
) -> None:
    """Set configuration for trade memory limits."""
    global MAX_LOSSES, SLIPPAGE_THRESHOLD, LOOKBACK_SECONDS
    MAX_LOSSES = max(1, int(max_losses))
    SLIPPAGE_THRESHOLD = float(slippage_threshold)
    LOOKBACK_SECONDS = max(0, int(lookback_seconds))


def _load() -> Dict[str, List[Dict[str, Any]]]:
    if LOG_FILE.exists():
        try:
            return json.loads(LOG_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save(data: Dict[str, List[Dict[str, Any]]]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(json.dumps(data))


def clear() -> None:
    """Remove the memory file."""
    if LOG_FILE.exists():
        LOG_FILE.unlink()


def record_loss(symbol: str, slippage: float) -> None:
    """Record a losing trade entry for ``symbol``."""
    data = _load()
    entry = {"ts": time.time(), "slippage": float(slippage)}
    data.setdefault(symbol, []).append(entry)
    _save(data)


def should_avoid(symbol: str) -> bool:
    """Return ``True`` if ``symbol`` should be avoided."""
    now = time.time()
    data = _load()
    entries = [e for e in data.get(symbol, []) if now - e.get("ts", 0) <= LOOKBACK_SECONDS]
    if any(e.get("slippage", 0.0) > SLIPPAGE_THRESHOLD for e in entries):
        return True
    return len(entries) >= MAX_LOSSES

