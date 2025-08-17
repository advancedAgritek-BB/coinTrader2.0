"""Minimal training stub for tests requiring cointrainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from cointrainer.io.csv7 import read_csv7


@dataclass
class TrainConfig:
    """Configuration placeholder used in tests."""

    symbol: str
    horizon: int = 0
    hold: float = 0.0
    n_estimators: int = 0


def train_from_csv7(path, cfg: TrainConfig) -> Tuple[object, dict]:
    """Dummy training routine.

    The function reads the CSV7 file to ensure the path is valid and returns a
    simple object along with metadata containing a synthetic feature list.  The
    heavy lifting from the real project is intentionally omitted.
    """

    if not isinstance(path, (str, bytes)):
        path = str(path)
    read_csv7(path)  # ensure the file can be parsed
    model = object()
    meta = {"feature_list": list(range(5))}
    return model, meta


__all__ = ["TrainConfig", "train_from_csv7"]

