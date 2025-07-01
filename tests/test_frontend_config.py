import yaml
from frontend import app


def test_set_and_load_execution_mode(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"execution_mode": "dry_run"}))
    monkeypatch.setattr(app, "CONFIG_FILE", cfg)
    app.set_execution_mode("live")
    assert app.load_execution_mode() == "live"
    with open(cfg) as f:
        data = yaml.safe_load(f)
    assert data["execution_mode"] == "live"
