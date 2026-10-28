import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from main import create_app
from model import calculate_risk
from schemas import EquipmentRequest
from services import get_risk_level


def post(client, data, **kwargs):
    response = client.post("/predict", json=data, **kwargs)
    assert response.status_code == 200, response.text
    return response.json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert float(r.headers["X-Process-Time"]) >= 0
    assert r.headers["X-Request-ID"]


def test_model_info(client):
    r = client.get("/model-info")
    assert r.json()["model_version"] == "1.0"
    assert r.json()["model_type"] == "rule-based-risk-scoring"


def test_predict_request_id_propagation(client, app, data):
    body = post(client, data, headers={"X-Request-ID": "trace-lab2"})
    assert body["request_id"] == "trace-lab2" and body["risk_score"] == 1
    with sqlite3.connect(app.state.settings.database_path) as db:
        row = db.execute("SELECT request_id, equipment_id, source FROM predictions").fetchone()
    assert row == ("trace-lab2", data["equipment_id"], "single")
    response = client.get("/predictions/trace-lab2", headers={"X-Request-ID": "new-get-id"})
    assert response.headers["X-Request-ID"] == "new-get-id"
    assert response.json() == body


def test_generated_id_matches_header(client, data):
    r = client.post("/predict", json=data)
    assert r.headers["X-Request-ID"] == r.json()["request_id"]
    assert len(r.json()["request_id"]) == 36
    assert r.json()["created_at"].endswith("Z")


@pytest.mark.parametrize("field,value", [
    ("temperature_c", 250), ("temperature_c", -41), ("vibration_mm_s", -1),
    ("pressure_bar", 51), ("operating_hours", -1), ("operating_hours", 1.5),
    ("days_since_maintenance", -1), ("previous_failures", -1), ("equipment_id", ""),
])
def test_pydantic_errors_have_id_and_no_storage(client, data, field, value):
    r = client.post("/predict", json={**data, field: value})
    assert r.status_code == 422
    assert r.json()["request_id"] == r.headers["X-Request-ID"]
    assert isinstance(r.json()["detail"], list)
    assert client.get("/predictions").json() == []


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_required_and_extra_fields(client, data, change):
    if change == "missing":
        data.pop("temperature_c")
    else:
        data["extra"] = 1
    assert client.post("/predict", json=data).status_code == 422


def test_business_error(client, data):
    r = client.post("/predict", json={**data, "equipment_type": "robot"})
    assert r.status_code == 400
    assert r.json()["request_id"] == r.headers["X-Request-ID"]
    assert client.get("/predictions").json() == []


def test_not_found(client):
    for path in ("/predictions/missing", "/requests/missing/predictions", "/missing-route"):
        r = client.get(path)
        assert r.status_code == 404 and r.json()["request_id"] == r.headers["X-Request-ID"]


def test_filter_order_limit(client, data, low_data):
    first = post(client, data)
    second = post(client, low_data)
    assert client.get("/predictions?limit=1").json() == [second]
    assert client.get("/predictions?risk_level=high").json() == [first]
    assert client.get("/predictions?risk_level=critical").status_code == 400
    assert client.get("/predictions?limit=-5").status_code == 422


def test_low_and_medium_scores(client, low_data):
    assert post(client, low_data)["risk_score"] == 0.05
    r = post(client, {**low_data, "temperature_c": 90, "vibration_mm_s": 8})
    assert r["risk_score"] == 0.55 and r["risk_level"] == "medium"


def test_exact_thresholds_and_categories(low_data):
    d = {**low_data, "temperature_c": 85, "vibration_mm_s": 7,
         "operating_hours": 10000, "pressure_bar": 12, "days_since_maintenance": 180}
    assert calculate_risk(EquipmentRequest(**d)) == 0.05
    assert [get_risk_level(v) for v in (0.29, 0.3, 0.69, 0.7)] == ["low", "medium", "medium", "high"]


def test_batch_scores_transaction_and_ambiguity(client, data, low_data):
    inputs = [data, low_data, {**low_data, "equipment_id": "MEDIUM", "temperature_c": 90}]
    r = client.post("/batch-predict", json={"items": inputs}, headers={"X-Request-ID": "batch-id"})
    assert r.status_code == 200
    body = r.json()
    assert body["batch_size"] == 3
    assert [x["risk_score"] for x in body["items"]] == [1.0, 0.05, 0.30]
    assert all(x["request_id"] == "batch-id" for x in body["items"])
    assert client.get("/requests/batch-id/predictions").json() == body["items"]
    assert client.get("/predictions/batch-id").status_code == 409
    assert client.get("/predictions/batch-id/explain").status_code == 409
    assert client.delete("/predictions/batch-id").status_code == 409


@pytest.mark.parametrize("n", [0, 9])
def test_batch_size(client, data, n):
    assert client.post("/batch-predict", json={"items": [data] * n}).status_code == 422
    assert client.get("/predictions").json() == []


def test_invalid_batch_is_rejected_before_write(client, data):
    r = client.post("/batch-predict", json={"items": [data, {**data, "equipment_type": "robot"}]})
    assert r.status_code == 400 and client.get("/predictions").json() == []


def test_single_item_batch_and_duplicate_id(client, data):
    r = client.post("/batch-predict", json={"items": [data]})
    rid = r.json()["request_id"]
    assert client.get(f"/predictions/{rid}").status_code == 200
    post(client, data, headers={"X-Request-ID": rid})
    assert client.get(f"/predictions/{rid}").status_code == 409
    assert len(client.get(f"/requests/{rid}/predictions").json()) == 2


def test_weekday_pagination(client, data):
    for i in range(5):
        post(client, {**data, "equipment_id": f"DEVICE-{i}"})
    weekday = datetime.now(timezone.utc).strftime("%A").lower()
    r = client.get("/predictions/by-weekday", params={"weekday": f" {weekday.upper()} ", "per_page": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["total_items"] == 5 and body["meta"]["total_pages"] == 3
    assert body["meta"]["prev_page"] is None
    second = client.get(body["meta"]["next_page"]).json()
    assert second["meta"]["current_page"] == 2 and second["meta"]["prev_page"]
    assert not ({x["equipment_id"] for x in body["data"]} & {x["equipment_id"] for x in second["data"]})
    assert client.get("/predictions/by-weekday", params={"weekday": weekday, "page": 4, "per_page": 2}).status_code == 404


@pytest.mark.parametrize("query,code", [
    ("weekday=invalid", 400), ("weekday=monday&page=-1", 422),
    ("weekday=monday&per_page=101", 422), ("weekday=monday&per_page=0", 422),
])
def test_weekday_validation(client, query, code):
    assert client.get("/predictions/by-weekday?" + query).status_code == code


def test_empty_pagination(client):
    r = client.get("/predictions/by-weekday?weekday=monday")
    assert r.status_code == 200
    assert r.json()["data"] == [] and r.json()["meta"]["total_pages"] == 0
    assert client.get("/predictions/by-weekday?weekday=monday&page=2").status_code == 404


def test_additional_endpoints(client, data):
    saved = post(client, data)
    rid = saved["request_id"]
    assert client.get(f"/equipment/{data['equipment_id']}/predictions").json() == [saved]
    explain = client.get(f"/predictions/{rid}/explain").json()
    assert explain["risk_score"] == saved["risk_score"] and len(explain["factors"]) == 6
    assert client.delete(f"/predictions/{rid}").status_code == 204
    assert client.get(f"/predictions/{rid}/explain").status_code == 404
    assert client.delete(f"/predictions/{rid}").status_code == 404


def test_survives_new_application_instance(settings, data):
    with TestClient(create_app(settings)) as c:
        saved = post(c, data)
    with TestClient(create_app(settings)) as c:
        assert c.get(f"/predictions/{saved['request_id']}").json() == saved
        assert c.get(f"/predictions/{saved['request_id']}/explain").status_code == 200


def test_sql_parameterization(client, data):
    rid = "id' OR 1=1 --"
    saved = post(client, data, headers={"X-Request-ID": rid})
    assert client.get("/predictions/" + rid).json() == saved
    assert client.get("/predictions/not-found' OR 1=1 --").status_code == 404


def test_model_unavailable(client, app, data):
    app.state.model.ready = False
    assert client.get("/model-info").json()["status"] == "unavailable"
    for path, body in [("/predict", data), ("/predict-queued", data), ("/batch-predict", {"items": [data]})]:
        assert client.post(path, json=body).status_code == 503
    assert client.get("/predictions").json() == []


def test_runtime_db_failure(client, app, data, tmp_path):
    app.state.storage.path = tmp_path / "missing" / "fail.db"
    for path in ("/predictions", "/predictions/missing"):
        assert client.get(path).status_code == 503
    assert client.post("/predict", json=data).status_code == 503


def test_unhandled_error_has_headers(client, app, data, monkeypatch):
    async def crash(*args, **kwargs):
        raise RuntimeError("private details")
    monkeypatch.setattr(app.state.storage, "save_many", crash)
    r = client.post("/predict", json=data)
    assert r.status_code == 500 and r.json()["detail"] == "Internal server error"
    assert r.json()["request_id"] == r.headers["X-Request-ID"]
    assert "private details" not in r.text and "X-Process-Time" in r.headers


def test_openapi_contract(client):
    schema = client.get("/openapi.json").json()
    for path in ("/predict", "/batch-predict", "/predict-queued", "/llm/chat", "/predict-with-llm"):
        assert "200" in schema["paths"][path]["post"]["responses"]
    assert "X-Request-ID" in str(schema["paths"]["/predict"]["post"]["parameters"]) or "x-request-id" in str(schema)


@pytest.mark.parametrize("equipment_type", ["motor", "pump", "compressor", "generator"])
def test_supported_equipment(client, data, equipment_type):
    assert post(client, {**data, "equipment_type": equipment_type})["risk_level"] == "high"
