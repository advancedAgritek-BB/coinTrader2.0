from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
PROGRESS_FILE = CACHE_DIR / "bootstrap_progress.json"


def reset_bootstrap_progress() -> None:
    """Remove any cached bootstrap progress file if present."""
    try:
        PROGRESS_FILE.unlink()
    except FileNotFoundError:
        pass
