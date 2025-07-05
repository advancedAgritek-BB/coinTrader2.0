from typing import Iterable, List


def compute_batches(items: Iterable[str], batch_size: int) -> List[List[str]]:
    """Split ``items`` into batches of size ``batch_size``."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    items_list = list(items)
    return [items_list[i : i + batch_size] for i in range(0, len(items_list), batch_size)]
