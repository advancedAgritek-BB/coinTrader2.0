from collections import deque
from typing import Iterable, List, Deque, Tuple


def compute_batches(items: Iterable[str], batch_size: int) -> List[List[str]]:
    """Split ``items`` into batches of size ``batch_size``."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    items_list = list(items)
    return [items_list[i : i + batch_size] for i in range(0, len(items_list), batch_size)]


def build_priority_queue(symbol_scores: List[Tuple[str, float]]) -> Deque[str]:
    """Return a queue of symbols sorted by descending score."""

    sorted_syms = sorted(symbol_scores, key=lambda x: x[1], reverse=True)
    return deque(sym for sym, _ in sorted_syms)
