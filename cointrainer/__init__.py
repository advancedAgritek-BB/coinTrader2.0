"""Expose modules from src/cointrainer as a top-level package."""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "src" / "cointrainer")]
