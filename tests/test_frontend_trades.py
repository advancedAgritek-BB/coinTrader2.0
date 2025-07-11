from frontend import app
import pandas as pd


def test_trades_data_route(tmp_path, monkeypatch):
    path = tmp_path / "trades.csv"
    df = pd.DataFrame([
        {"symbol": "XBT/USDT", "side": "buy", "amount": 1, "price": 100, "timestamp": "t"},
        {"symbol": "ETH/USDT", "side": "sell", "amount": 2, "price": 200, "timestamp": "t2"},
    ])
    df.to_csv(path, index=False, header=False)
    monkeypatch.setattr(app, "TRADE_FILE", path)
    client = app.app.test_client()
    resp = client.get("/trades_data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 2
    assert data[0]["symbol"] == "XBT/USDT"
