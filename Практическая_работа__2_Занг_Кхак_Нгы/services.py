"""Справочники, бизнес-валидация и асинхронный инференс."""

import asyncio
import logging
import math
from datetime import datetime, timezone

from fastapi import HTTPException, status
from schemas import EquipmentRequest, PredictionResponse, RiskLevel
from model import RiskModel

logger = logging.getLogger("equipment-risk.services")

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


def build_prediction(data: EquipmentRequest, score: float, request_id: str) -> PredictionResponse:
    level = get_risk_level(score)
    return PredictionResponse(
        request_id=request_id,
        equipment_id=data.equipment_id,
        risk_score=score,
        risk_level=level,
        recommendation=get_recommendation(level),
        model_version=MODEL_VERSION,
        created_at=datetime.now(timezone.utc),
    )


async def run_inference(model: RiskModel, items: list[EquipmentRequest], request_ids: list[str]):
    if not model.ready:
        raise HTTPException(503, "Model is temporarily unavailable")
    logger.info("Inference started | request_ids=%s | batch_size=%s", request_ids, len(items))
    try:
        scores = await asyncio.wait_for(
            model.model_predict_batch(items),
            timeout=model.settings.inference_timeout_seconds,
        )
        if len(scores) != len(items):
            raise ValueError("Model returned an unexpected number of scores")
        if any(isinstance(s, bool) or not isinstance(s, (int, float))
               or not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError("Model scores must be finite numbers between 0 and 1")
    except asyncio.TimeoutError as exc:
        logger.warning("Inference timeout | request_ids=%s", request_ids)
        raise HTTPException(504, "Inference timeout") from exc
    except Exception as exc:
        logger.exception("Model failure | request_ids=%s", request_ids)
        raise HTTPException(503, "Model is temporarily unavailable") from exc
    logger.info("Inference completed | request_ids=%s | batch_size=%s", request_ids, len(items))
    return scores
