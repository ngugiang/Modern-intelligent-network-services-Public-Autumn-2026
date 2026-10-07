"""Автоматическая проверка обязательных и дополнительных endpoints."""

import pytest
from fastapi.testclient import TestClient

import main
from main import app
from model import calculate_risk, get_risk_factors
from schemas import EquipmentRequest
from services import get_risk_level
from storage import clear_storage, predictions, source_requests


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_state(monkeypatch):
    clear_storage()
    monkeypatch.setattr(main, "MODEL_READY", True)
    yield
    clear_storage()


@pytest.fixture
def valid_data():
    return {
        "equipment_id": "MOTOR-001",
        "equipment_type": "motor",
        "temperature_c": 92.5,
        "vibration_mm_s": 8.4,
        "operating_hours": 12000,
        "pressure_bar": 14,
        "days_since_maintenance": 220,
        "previous_failures": 2,
    }


def post_valid(data):
    response = client.post("/predict", json=data)
    assert response.status_code == 201
    return response.json()


# 10 обязательных сценариев из методических указаний.

def test_01_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_02_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert r.json()["model_version"] == "1.0"


def test_03_predict_success(valid_data):
    r = client.post("/predict", json=valid_data)
    assert r.status_code == 201
    assert r.json()["risk_score"] == 1.0
    assert r.json()["risk_level"] == "high"
    assert r.json()["request_id"]
    assert r.json()["created_at"]


def test_04_pydantic_invalid_temperature(valid_data):
    valid_data["temperature_c"] = 250
    r = client.post("/predict", json=valid_data)
    assert r.status_code == 422
    assert len(predictions) == 0


def test_05_unknown_equipment_type(valid_data):
    valid_data["equipment_type"] = "robot"
    r = client.post("/predict", json=valid_data)
    assert r.status_code == 400
    assert "Unknown equipment type" in r.json()["detail"]


def test_06_existing_request_id(valid_data):
    item = post_valid(valid_data)
    r = client.get(f"/predictions/{item['request_id']}")
    assert r.status_code == 200
    assert r.json() == item


def test_07_nonexistent_request_id():
    assert client.get("/predictions/unknown-id").status_code == 404


def test_08_limit_2(valid_data):
    for n in range(3):
        post_valid({**valid_data, "equipment_id": f"MOTOR-{n}"})
    r = client.get("/predictions?limit=2")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_09_filter_high(valid_data):
    post_valid(valid_data)
    post_valid({**valid_data, "temperature_c": 50, "vibration_mm_s": 2,
                "operating_hours": 100, "pressure_bar": 2,
                "days_since_maintenance": 10, "previous_failures": 0})
    r = client.get("/predictions?risk_level=high")
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert all(x["risk_level"] == "high" for x in r.json())


def test_10_invalid_limit():
    assert client.get("/predictions?limit=-5").status_code == 422


# Расширенные сценарии.

def test_11_filter_rejects_unknown_level():
    r = client.get("/predictions?risk_level=critical")
    assert r.status_code == 400


def test_12_no_predictions():
    r = client.get("/predictions")
    assert r.status_code == 200 and r.json() == []


def test_13_model_unavailable(monkeypatch, valid_data):
    monkeypatch.setattr(main, "MODEL_READY", False)
    assert client.get("/model-info").json()["status"] == "unavailable"
    assert client.post("/predict", json=valid_data).status_code == 503
    assert predictions == {}


def test_14_missing_field(valid_data):
    valid_data.pop("vibration_mm_s")
    assert client.post("/predict", json=valid_data).status_code == 422


def test_15_extra_field(valid_data):
    valid_data["unexpected"] = 123
    assert client.post("/predict", json=valid_data).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("temperature_c", -41),
    ("vibration_mm_s", -1),
    ("pressure_bar", 51),
    ("operating_hours", -1),
    ("days_since_maintenance", -1),
    ("previous_failures", -1),
    ("equipment_id", ""),
    ("operating_hours", 1.5),
])
def test_16_field_validation(valid_data, field, value):
    valid_data[field] = value
    assert client.post("/predict", json=valid_data).status_code == 422


def test_17_low_risk(valid_data):
    data = {**valid_data, "temperature_c": 50, "vibration_mm_s": 2,
            "operating_hours": 100, "pressure_bar": 2,
            "days_since_maintenance": 10, "previous_failures": 0}
    r = post_valid(data)
    assert r["risk_score"] == 0.05 and r["risk_level"] == "low"


def test_18_medium_risk(valid_data):
    data = {**valid_data, "temperature_c": 90, "vibration_mm_s": 8,
            "operating_hours": 100, "pressure_bar": 2,
            "days_since_maintenance": 10, "previous_failures": 0}
    r = post_valid(data)
    assert r["risk_score"] == 0.55 and r["risk_level"] == "medium"


def test_19_exact_thresholds_not_triggered(valid_data):
    data = {**valid_data, "temperature_c": 85, "vibration_mm_s": 7,
            "operating_hours": 10000, "pressure_bar": 12,
            "days_since_maintenance": 180, "previous_failures": 0}
    assert post_valid(data)["risk_score"] == 0.05


def test_20_risk_level_boundaries():
    assert get_risk_level(0.29) == "low"
    assert get_risk_level(0.3) == "medium"
    assert get_risk_level(0.69) == "medium"
    assert get_risk_level(0.7) == "high"


def test_21_explanation(valid_data):
    item = post_valid(valid_data)
    r = client.get(f"/predictions/{item['request_id']}/explain")
    assert r.status_code == 200
    body = r.json()
    assert len(body["factors"]) == 6
    assert sum(x["contribution"] for x in body["factors"]) > 0
    assert body["risk_score"] == item["risk_score"]


def test_22_explanation_not_found():
    assert client.get("/predictions/missing/explain").status_code == 404


def test_23_equipment_history(valid_data):
    post_valid(valid_data)
    post_valid({**valid_data, "equipment_id": "PUMP-002", "equipment_type": "pump"})
    post_valid(valid_data)
    r = client.get("/equipment/MOTOR-001/predictions")
    assert r.status_code == 200
    assert len(r.json()) == 2
    assert all(x["equipment_id"] == "MOTOR-001" for x in r.json())


def test_24_equipment_history_missing():
    r = client.get("/equipment/MISSING/predictions")
    assert r.status_code == 200 and r.json() == []


def test_25_delete(valid_data):
    item = post_valid(valid_data)
    request_id = item["request_id"]
    r = client.delete(f"/predictions/{request_id}")
    assert r.status_code == 204 and r.content == b""
    assert request_id not in source_requests
    assert client.get(f"/predictions/{request_id}").status_code == 404
    assert client.get(f"/predictions/{request_id}/explain").status_code == 404


def test_26_delete_missing():
    assert client.delete("/predictions/missing").status_code == 404


def test_27_newest_first(valid_data):
    one = post_valid({**valid_data, "equipment_id": "FIRST"})
    two = post_valid({**valid_data, "equipment_id": "SECOND"})
    r = client.get("/predictions")
    assert [x["request_id"] for x in r.json()] == [two["request_id"], one["request_id"]]


def test_28_middleware_header():
    r = client.get("/health")
    assert r.status_code == 200
    assert float(r.headers["X-Process-Time"]) >= 0


def test_29_openapi_contract():
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    for path in ("/health", "/model-info", "/predict", "/predictions",
                 "/predictions/{request_id}"):
        assert path in paths
    assert "201" in paths["/predict"]["post"]["responses"]


def test_30_supported_types(valid_data):
    for t in ("motor", "pump", "compressor", "generator"):
        assert post_valid({**valid_data, "equipment_type": t})["risk_level"] == "high"


def test_31_rule_breakdown_and_score(valid_data):
    data = EquipmentRequest(**valid_data)
    factors = get_risk_factors(data)
    assert len(factors) == 6
    assert calculate_risk(data) == min(1.0, round(0.05 + sum(x.contribution for x in factors), 2))


def test_32_response_has_request_id_and_time(valid_data):
    r = post_valid(valid_data)
    assert isinstance(r["request_id"], str) and len(r["request_id"]) == 36
    assert r["created_at"].endswith("Z") or "+00:00" in r["created_at"]
