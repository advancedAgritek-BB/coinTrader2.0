from pathlib import Path
import json
from typing import Dict

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
PROGRESS_FILE = CACHE_DIR / "bootstrap_progress.json"


def reset_bootstrap_progress() -> None:
    """Remove any cached bootstrap progress file if present."""
    try:
        PROGRESS_FILE.unlink()
    except FileNotFoundError:
        pass


def update_bootstrap_progress(stats: Dict[str, Dict[str, int]]) -> None:
    """Merge *stats* into the cached bootstrap progress file.

    Parameters
    ----------
    stats : Dict[str, Dict[str, int]]
        Mapping of timeframe -> {"fetched": int, "required": int}
    """

    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with PROGRESS_FILE.open() as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    data.update(stats)
    with PROGRESS_FILE.open("w") as f:
        json.dump(data, f)
