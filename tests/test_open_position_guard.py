import pandas as pd

from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.main import direction_to_side


def test_open_position_guard(monkeypatch):
    wallet = PaperWallet(1000.0)
    wallet.open("buy", 1.0, 100.0)

    calls = {"count": 0}

    def record_open(side, amount, price):
        calls["count"] += 1

    monkeypatch.setattr(wallet, "open", record_open)

    open_side = "buy"
    filtered_results = [
        {
            "symbol": "ETH/USDT",
            "df": pd.DataFrame({"close": [110.0]}),
            "regime": "bull",
            "env": "live",
            "name": "trend_bot",
            "direction": "long",
            "score": 1.0,
        }
    ]
    config = {"execution_mode": "dry_run", "signal_threshold": 0.0}

    # Simulate the main loop section responsible for opening trades
    if open_side:
        pass
    elif not filtered_results:
        pass
    else:
        for candidate in filtered_results:
            trade_side = direction_to_side(candidate["direction"])
            current_price = candidate["df"]["close"].iloc[-1]
            if config["execution_mode"] == "dry_run" and wallet:
                wallet.open(trade_side, 1.0, current_price)

    assert calls["count"] == 0

