"""Pydantic: входные и выходные схемы REST API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from config import MAX_BATCH_SIZE


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


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[EquipmentRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchPredictionResponse(BaseModel):
    request_id: str
    batch_size: int
    processing_time_ms: float
    items: list[PredictionResponse]


class PaginationMeta(BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    next_page: str | None
    prev_page: str | None


class PaginatedPredictionsResponse(BaseModel):
    data: list[PredictionResponse]
    meta: PaginationMeta


class QueueStatusResponse(BaseModel):
    queue_size: int
    queue_max_size: int
    max_batch_size: int
    max_batch_wait_ms: float
    inference_timeout_seconds: float


class LLMChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=10000, pattern=r"\S")
    model: str = Field(default="auto", min_length=1, max_length=200, pattern=r"\S")
    temperature: float = Field(default=0.3, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(default=300, ge=1, le=4096)


class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: str | None = Field(default=None, min_length=1)
    answer: str = Field(min_length=1, pattern=r"\S")
    finish_reason: str | None
    latency_ms: float = Field(ge=0)


class PredictWithLLMResponse(BaseModel):
    prediction: PredictionResponse
    llm: LLMChatResponse


class ErrorResponse(BaseModel):
    detail: str | list[dict]
    request_id: str | None
