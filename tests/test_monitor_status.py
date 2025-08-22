from types import SimpleNamespace

from crypto_bot.main import SessionState, format_monitor_line


def test_monitor_line_shows_balance_and_log_only():
    """format_monitor_line should now emit only balance and last log."""
    ctx = SimpleNamespace(active_universe=["BTC/USD", "ETH/USD"], config={"timeframes": ["1h", "4h"]})
    session_state = SessionState()
    balance = 1000.0
    positions = {}
    last_log = "test"

    line = format_monitor_line(ctx, session_state, balance, positions, last_log)

    assert line == "[Monitor] balance=$1,000.00 last='test'"
