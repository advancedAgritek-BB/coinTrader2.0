from __future__ import annotations

"""Simple console monitor for displaying trades."""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from . import log_reader

TRADE_FILE = Path("crypto_bot/logs/trades.csv")


def run(exchange: Any | None = None, wallet: Any | None = None, trade_file: Path = TRADE_FILE) -> str:
    """Read trades from ``trade_file`` and print them as a table.

    Returns the rendered table as text so tests can verify the output.
    """
    df = log_reader._read_trades(trade_file)
    console = Console(record=True)
    table = Table(show_header=True, header_style="bold")
    table.add_column("symbol")
    table.add_column("side")
    table.add_column("amount")
    table.add_column("price")

    for _, row in df.iterrows():
        table.add_row(
            str(row.get("symbol", "")),
            str(row.get("side", "")),
            str(row.get("amount", "")),
            str(row.get("price", "")),
        )

    console.print(table)
    return console.export_text()
