from frontend import app


def test_dashboard_route(tmp_path, monkeypatch):
    trade_file = tmp_path / "trades.csv"
    trade_file.write_text("XBT/USDT,buy,1,100\nXBT/USDT,sell,1,110")
    monkeypatch.setattr(app, "TRADE_FILE", trade_file)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("strategy_allocation:\n  trend_bot: 0.5\n  grid_bot: 0.5")
    monkeypatch.setattr(app, "CONFIG_FILE", cfg)
    regime_file = tmp_path / "regime.txt"
    regime_file.write_text("trending\nsideways")
    monkeypatch.setattr(app, "REGIME_FILE", regime_file)
    client = app.app.test_client()
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Total PnL" in resp.data


def test_static_resources_load():
    client = app.app.test_client()
    resp = client.get('/static/styles.css')
    assert resp.status_code == 200
    assert b'font-family' in resp.data
