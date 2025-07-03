import pandas as pd
from crypto_bot.risk.exit_manager import should_exit, calculate_trailing_stop

def _sample_df(length=30):
    data = {
        "open": list(range(length)),
        "high": list(range(length)),
        "low": list(range(length)),
        "close": list(range(length)),
        "volume": [10] * length,
    }
    return pd.DataFrame(data)

def test_should_exit_no_valueerror_with_df_current():
    df_current = _sample_df()
    best_df = _sample_df()
    current_price = df_current["close"].iloc[-1]
    trailing_stop = calculate_trailing_stop(df_current["close"], 0.1)
    config = {"exit_strategy": {"trailing_stop_pct": 0.1}, "symbol": "TEST"}

    df_to_use = df_current if df_current is not None else best_df
    exit_signal, new_stop = should_exit(
        df_to_use,
        current_price,
        trailing_stop,
        config,
        None,
    )

    assert isinstance(exit_signal, bool)
    assert isinstance(new_stop, float)

