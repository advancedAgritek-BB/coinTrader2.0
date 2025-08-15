from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd


@dataclass
class TrainConfig:
    symbol: str
    horizon: int
    hold: float
    n_estimators: int


def train_from_csv7(path, cfg: TrainConfig) -> Tuple[object, Dict[str, Any]]:
    """Dummy training routine used for tests.

    Loads the CSV file to ensure the path is valid and returns a dummy model
    together with minimal metadata including a ``feature_list`` entry.
    """
    # Simply read the file to validate input; ignore contents
    pd.read_csv(path, header=None)
    model = object()
    meta: Dict[str, Any] = {"feature_list": ["f1", "f2", "f3", "f4", "f5"]}
    return model, meta
