import yaml
from frontend import utils


def test_set_and_load_execution_mode(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"execution_mode": "dry_run"}))
    utils.set_execution_mode("live", cfg)
    assert utils.load_execution_mode(cfg) == "live"
    with open(cfg) as f:
        data = yaml.safe_load(f)
    assert data["execution_mode"] == "live"
