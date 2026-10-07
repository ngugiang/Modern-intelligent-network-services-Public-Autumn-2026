"""Демонстрационное in-memory хранилище. Данные теряются при перезапуске."""

from schemas import EquipmentRequest

# Ключ — request_id. Храним результат и исходный запрос для объяснения.
predictions: dict[str, dict] = {}
source_requests: dict[str, EquipmentRequest] = {}


def save_prediction(result: dict, data: EquipmentRequest) -> None:
    request_id = result["request_id"]
    predictions[request_id] = result.copy()
    source_requests[request_id] = data.model_copy(deep=True)


def get_prediction(request_id: str) -> dict | None:
    return predictions.get(request_id)


def get_source_request(request_id: str) -> EquipmentRequest | None:
    return source_requests.get(request_id)


def list_predictions(limit: int, risk_level: str | None = None) -> list[dict]:
    # Новые записи возвращаем первыми.
    results = list(reversed(list(predictions.values())))
    if risk_level is not None:
        results = [item for item in results if item["risk_level"] == risk_level]
    return results[:limit]


def get_equipment_predictions(equipment_id: str) -> list[dict]:
    return [
        item for item in reversed(list(predictions.values()))
        if item["equipment_id"] == equipment_id
    ]


def delete_prediction(request_id: str) -> bool:
    if request_id not in predictions:
        return False
    predictions.pop(request_id)
    source_requests.pop(request_id, None)
    return True


def clear_storage() -> None:
    """Очистка данных для изолированных автоматических тестов."""
    predictions.clear()
    source_requests.clear()
