from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Mapping, Tuple, List

import pandas as pd

from . import grid_bot


def generate_signals(
    df_map: Mapping[str, pd.DataFrame],
    config: grid_bot.ConfigType | None = None,
) -> List[Tuple[str, float, str]]:
    """Check correlated pairs for price discrepancies and trigger grid signals.

    Parameters
    ----------
    df_map : Mapping[str, pd.DataFrame]
        Mapping of symbol to OHLCV DataFrame.
    config : GridConfig or dict, optional
        Configuration containing ``arbitrage_pairs`` and ``arbitrage_threshold``.

    Returns
    -------
    list[tuple[str, float, str]]
        ``[(symbol, score, direction), ...]`` for each pair when arbitrage is
        detected. Returns an empty list when no opportunity is found.
    """
    cfg = grid_bot.GridConfig.from_dict(config if isinstance(config, dict) else asdict(config) if config else {})
    pairs: Iterable[Tuple[str, str]] = getattr(cfg, "arbitrage_pairs", [])
    threshold: float = getattr(cfg, "arbitrage_threshold", 0.005)

    results: List[Tuple[str, float, str]] = []
    for sym_a, sym_b in pairs:
        df_a = df_map.get(sym_a)
        df_b = df_map.get(sym_b)
        if df_a is None or df_b is None or df_a.empty or df_b.empty:
            continue
        price_a = float(df_a["close"].iloc[-1])
        price_b = float(df_b["close"].iloc[-1])
        if price_b == 0:
            continue
        spread = abs(price_a - price_b) / price_b
        if spread < threshold:
            continue
        score_a, dir_a = grid_bot.generate_signal(df_a, config={**asdict(cfg), "symbol": sym_a})
        score_b, dir_b = grid_bot.generate_signal(df_b, config={**asdict(cfg), "symbol": sym_b})
        results.append((sym_a, score_a, dir_a))
        results.append((sym_b, score_b, dir_b))
    return results
