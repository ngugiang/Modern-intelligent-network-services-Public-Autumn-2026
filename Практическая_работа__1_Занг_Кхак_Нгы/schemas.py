"""Pydantic: входные и выходные схемы REST API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EquipmentRequest(BaseModel):
    """Показатели промышленного оборудования для оценки риска."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "equipment_id": "MOTOR-001",
                "equipment_type": "motor",
                "temperature_c": 92.5,
                "vibration_mm_s": 8.4,
                "operating_hours": 12000,
                "pressure_bar": 14.0,
                "days_since_maintenance": 220,
                "previous_failures": 2,
            }
        },
    )

    equipment_id: str = Field(min_length=1, max_length=50, description="ID оборудования")
    equipment_type: str = Field(min_length=1, description="Тип: motor, pump, compressor, generator")
    temperature_c: float = Field(ge=-40, le=200, allow_inf_nan=False, description="Температура, °C")
    vibration_mm_s: float = Field(ge=0, le=100, allow_inf_nan=False, description="Вибрация, мм/с")
    operating_hours: int = Field(ge=0, description="Общее время работы, часы")
    pressure_bar: float = Field(ge=0, le=50, allow_inf_nan=False, description="Давление, бар")
    days_since_maintenance: int = Field(ge=0, description="Дней после обслуживания")
    previous_failures: int = Field(ge=0, description="Число предыдущих отказов")


RiskLevel = Literal["low", "medium", "high"]


class PredictionResponse(BaseModel):
    request_id: str
    equipment_id: str
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    recommendation: str
    model_version: str
    created_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: Literal["ready", "unavailable"]


class RiskFactor(BaseModel):
    factor: str
    threshold: str
    is_triggered: bool
    contribution: float = Field(ge=0, le=1)


class ExplainResponse(BaseModel):
    request_id: str
    equipment_id: str
    base_score: float
    factors: list[RiskFactor]
    risk_score: float
    explanation: str
