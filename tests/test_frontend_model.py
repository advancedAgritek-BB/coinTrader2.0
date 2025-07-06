import json
from io import BytesIO
from frontend import app


def test_model_page(tmp_path, monkeypatch):
    metrics = {"accuracy": 0.8, "auc": 0.9, "trained_at": "2023-01-01"}
    report = tmp_path / "report.json"
    report.write_text(json.dumps(metrics))
    monkeypatch.setattr(app, "MODEL_REPORT", report)
    client = app.app.test_client()
    resp = client.get("/model")
    assert resp.status_code == 200
    assert b"0.8" in resp.data


def test_train_route(monkeypatch, tmp_path):
    called = {}

    def dummy(path):
        called["path"] = path

    monkeypatch.setattr(app.ml, "train_from_csv", dummy)
    monkeypatch.setattr(app, "MODEL_REPORT", tmp_path / "report.json")
    client = app.app.test_client()
    data = {"csv": (BytesIO(b"a,b\n1,2"), "data.csv")}
    resp = client.post(
        "/train_model",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert called


def test_validate_route(monkeypatch, tmp_path):
    metrics = {"accuracy": 1.0, "auc": 1.0, "trained_at": "now"}

    def dummy(path):
        return metrics

    monkeypatch.setattr(app.ml, "validate_from_csv", dummy)
    report = tmp_path / "report.json"
    monkeypatch.setattr(app, "MODEL_REPORT", report)
    client = app.app.test_client()
    data = {"csv": (BytesIO(b"a,b\n1,2"), "data.csv")}
    resp = client.post(
        "/validate_model",
        data=data,
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert json.loads(report.read_text()) == metrics

