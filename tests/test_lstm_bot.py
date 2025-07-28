import pandas as pd
from crypto_bot.strategy import lstm_bot


def make_df():
    return pd.DataFrame({
        "open": [1, 1.1, 1.2],
        "high": [1.1, 1.2, 1.3],
        "low": [0.9, 1.0, 1.1],
        "close": [1.0, 1.1, 1.2],
        "volume": [100, 100, 100],
    })


def test_generate_signal_long(monkeypatch):
    df = make_df()

    class DummyModel:
        def predict(self, data):
            current = float(data["close"].iloc[-1])
            return [current * 1.02]

    monkeypatch.setattr(lstm_bot, "MODEL", DummyModel())
    score, direction = lstm_bot.generate_signal(df)
    assert direction == "long"
    assert score > 0


def test_generate_signal_none_without_model(monkeypatch):
    df = make_df()
    monkeypatch.setattr(lstm_bot, "MODEL", None)
    assert lstm_bot.generate_signal(df) == (0.0, "none")
