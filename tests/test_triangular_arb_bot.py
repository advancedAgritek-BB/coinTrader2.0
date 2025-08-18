import importlib.util
from pathlib import Path

import pandas as pd


path = Path(__file__).resolve().parents[1] / "crypto_bot/strategy/triangular_arb_bot.py"
spec = importlib.util.spec_from_file_location("triangular_arb_bot", path)
triangular_arb_bot = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(triangular_arb_bot)


def make_df() -> pd.DataFrame:
    data = {
        "high": [1.0] * 20,
        "low": [1.0] * 20,
        "close": [1.0] * 20,
    }
    return pd.DataFrame(data)


def test_triangular_arb_signal(monkeypatch):
    async def fake_fetch(_ex, pair):
        data = {
            "A/B": {"bids": [[2.0]], "asks": [[2.1]]},
            "B/C": {"bids": [[3.0]], "asks": [[3.1]]},
            "A/C": {"bids": [[5.0]], "asks": [[5.0]]},
        }
        return data[pair]

    monkeypatch.setattr(triangular_arb_bot, "fetch_order_book_async", fake_fetch)
    df = make_df()
    cfg = {
        "triangular_arb_bot": {
            "arb_pairs": [("A/B", "B/C", "A/C")],
            "spread_threshold": 0.005,
            "fee_rate": 0.0,
        }
    }
    score, direction, atr = triangular_arb_bot.generate_signal(df, cfg, exchange=object())
    assert direction == "long"
    assert score > 0
    assert atr >= 0


def test_triangular_arb_no_signal(monkeypatch):
    async def fake_fetch(_ex, pair):
        data = {
            "A/B": {"bids": [[2.0]], "asks": [[2.1]]},
            "B/C": {"bids": [[3.0]], "asks": [[3.1]]},
            "A/C": {"bids": [[6.1]], "asks": [[6.1]]},
        }
        return data[pair]

    monkeypatch.setattr(triangular_arb_bot, "fetch_order_book_async", fake_fetch)
    df = make_df()
    cfg = {
        "triangular_arb_bot": {
            "arb_pairs": [("A/B", "B/C", "A/C")],
            "spread_threshold": 0.005,
            "fee_rate": 0.0,
        }
    }
    score, direction, atr = triangular_arb_bot.generate_signal(df, cfg, exchange=object())
    assert (score, direction) == (0.0, "none")
    assert atr == 0

