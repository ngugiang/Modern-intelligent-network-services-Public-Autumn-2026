"""Справочники, бизнес-валидация и постобработка."""

from fastapi import HTTPException, status
from schemas import EquipmentRequest, RiskLevel

MODEL_NAME = "equipment-failure-risk-model"
MODEL_VERSION = "1.0"
MODEL_TYPE = "rule-based-risk-scoring"
ALLOWED_EQUIPMENT_TYPES = {"motor", "pump", "compressor", "generator"}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}


def validate_equipment(data: EquipmentRequest) -> None:
    """Проверки предметной области: ошибка 400, не 422 Pydantic."""
    if data.equipment_type not in ALLOWED_EQUIPMENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown equipment type: {data.equipment_type}. "
                f"Allowed: {sorted(ALLOWED_EQUIPMENT_TYPES)}"
            ),
        )


def validate_risk_filter(level: str | None) -> None:
    if level is not None and level not in ALLOWED_RISK_LEVELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="risk_level must be 'low', 'medium' or 'high'",
        )


def get_risk_level(score: float) -> RiskLevel:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def get_recommendation(level: RiskLevel) -> str:
    if level == "low":
        return "Продолжить плановое техническое обслуживание"
    if level == "medium":
        return "Назначить внеплановую диагностику оборудования"
    return "Высокий риск: остановить оборудование и провести осмотр специалистом"
