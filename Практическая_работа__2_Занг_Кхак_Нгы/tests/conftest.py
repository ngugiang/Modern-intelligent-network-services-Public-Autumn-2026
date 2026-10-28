import httpx
import pytest
from fastapi.testclient import TestClient

from config import Settings
from main import create_app


@pytest.fixture
def data():
    return {
        "equipment_id": "MOTOR-001", "equipment_type": "motor", "temperature_c": 92.5,
        "vibration_mm_s": 8.4, "operating_hours": 12000, "pressure_bar": 14.0,
        "days_since_maintenance": 220, "previous_failures": 2,
    }


@pytest.fixture
def low_data(data):
    return {**data, "equipment_id": "PUMP-LOW", "temperature_c": 50, "vibration_mm_s": 2,
            "operating_hours": 100, "pressure_bar": 2, "days_since_maintenance": 10, "previous_failures": 0}


def llm_ok(request):
    # Фикстура не выдаётся за ответ настоящего провайдера.
    return httpx.Response(200, json={
        "model": "test-model", "choices": [{"message": {"content": "Тестовое объяснение риска."},
                                               "finish_reason": "stop"}],
    })


@pytest.fixture
def settings(tmp_path):
    return Settings(database_path=tmp_path / "test.db")


@pytest.fixture
def app(settings):
    return create_app(settings, llm_transport=httpx.MockTransport(llm_ok))


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c
