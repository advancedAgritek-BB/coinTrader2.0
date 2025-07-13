import asyncio
import json
from pathlib import Path
from typing import List

import pandas as pd

from crypto_bot.solana import get_solana_new_tokens
from crypto_bot.strategies import sniper_solana

DATA_FILE = Path(__file__).resolve().parent / "sample_solana_trades.json"


def load_data(path: Path) -> pd.DataFrame:
    """Return OHLCV data from ``path``."""
    with open(path) as fh:
        data = json.load(fh)
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def backtest(df: pd.DataFrame, *, atr_window: int, min_volume_usd: float) -> dict:
    """Run a simple sniper simulation over ``df``."""
    cfg = {"atr_window": atr_window}
    pnl = 0.0
    wins = 0
    trades = 0
    entry_price = None
    position: str | None = None

    for i in range(len(df)):
        row = df.iloc[: i + 1]
        if row["volume"].iloc[-1] < min_volume_usd:
            continue
        score, direction = sniper_solana.generate_signal(row, cfg)
        if direction in {"long", "short"} and entry_price is None:
            entry_price = row["close"].iloc[-1]
            position = direction
            trades += 1
        elif direction == "close" and entry_price is not None:
            exit_price = row["close"].iloc[-1]
            change = (exit_price - entry_price) / entry_price
            if position == "short":
                change = -change
            pnl += change
            wins += change > 0
            entry_price = None
            position = None

    if entry_price is not None:
        exit_price = df["close"].iloc[-1]
        change = (exit_price - entry_price) / entry_price
        if position == "short":
            change = -change
        pnl += change
        wins += change > 0

    roi = pnl / trades if trades else 0.0
    win_rate = wins / trades if trades else 0.0
    return {
        "atr_window": atr_window,
        "min_volume_usd": min_volume_usd,
        "pnl": pnl,
        "roi": roi,
        "win_rate": win_rate,
    }


def main() -> None:
    df = load_data(DATA_FILE)
    tokens: List[str] = asyncio.run(get_solana_new_tokens({}))
    print(f"Loaded {len(tokens)} tokens from scanner")
    results = []
    for vol in [0, 1200, 1500]:
        for atr in [10, 14, 20]:
            results.append(backtest(df, atr_window=atr, min_volume_usd=vol))
    res_df = pd.DataFrame(results)
    print(res_df)
    best = res_df.sort_values("pnl", ascending=False).iloc[0]
    print("Best config:")
    print(best)


if __name__ == "__main__":
    main()
