import time
from types import SimpleNamespace

import pandas as pd

from crypto_bot.main import SessionState, format_monitor_line
from crypto_bot.utils import market_loader


def test_monitor_line_includes_metrics():
    ctx = SimpleNamespace(active_universe=["BTC/USD", "ETH/USD"], config={"timeframes": ["1h", "4h"]})
    session_state = SessionState()
    session_state.df_cache["1h"] = {"BTC/USD": pd.DataFrame([1], index=[pd.Timestamp("2024-01-01")])}
    balance = 1000.0
    positions = {}
    last_log = "test"
    market_loader.IO_TIMESTAMPS.clear()
    market_loader.record_io()
    market_loader.record_io()
    line = format_monitor_line(ctx, session_state, balance, positions, last_log)
    assert "OHLCV 1h: 1/2 | 4h: 0/2" in line
    assert "IOPS" in line and "/s" in line
