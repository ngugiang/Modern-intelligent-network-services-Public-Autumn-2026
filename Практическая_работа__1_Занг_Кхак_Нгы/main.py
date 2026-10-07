"""REST API оценки риска отказа оборудования на FastAPI."""

import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse

from model import BASE_SCORE, calculate_risk, get_risk_factors
from schemas import (
    EquipmentRequest,
    ExplainResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from services import (
    MODEL_NAME,
    MODEL_TYPE,
    MODEL_VERSION,
    get_recommendation,
    get_risk_level,
    validate_equipment,
    validate_risk_filter,
)
from storage import (
    delete_prediction,
    get_equipment_predictions,
    get_prediction,
    get_source_request,
    list_predictions,
    save_prediction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("equipment-risk")

MODEL_READY = True  # False -> POST /predict отвечает HTTP 503.

app = FastAPI(
    title="Equipment Failure Risk API",
    description="Учебный REST API для оценки риска отказа промышленного оборудования",
    version="1.0.0",
    openapi_tags=[
        {"name": "Monitoring", "description": "Состояние API и модели"},
        {"name": "Predictions", "description": "Оценка и сохранение риска"},
        {"name": "Additional", "description": "Расширенные возможности"},
    ],
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(round(time.perf_counter() - start, 6))
    return response


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled error at %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get(
    "/health", response_model=HealthResponse, tags=["Monitoring"],
    summary="Проверка работоспособности API",
)
def health():
    return {"status": "ok"}


@app.get(
    "/model-info", response_model=ModelInfoResponse, tags=["Monitoring"],
    summary="Информация о модели и готовности",
)
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "status": "ready" if MODEL_READY else "unavailable",
    }


@app.post(
    "/predict", response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED, tags=["Predictions"],
    summary="Оценить риск отказа оборудования",
    responses={400: {"description": "Неизвестный тип оборудования"},
               503: {"description": "Модель недоступна"}},
)
def predict(data: EquipmentRequest):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model is temporarily unavailable")

    # Pydantic автоматически проверяет типы и диапазоны (422).
    # Бизнес-правила проверяются отдельно (400).
    validate_equipment(data)

    score = calculate_risk(data)
    level = get_risk_level(score)
    result = {
        "request_id": str(uuid.uuid4()),
        "equipment_id": data.equipment_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": get_recommendation(level),
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc),
    }
    save_prediction(result, data)
    logger.info("Prediction completed | id=%s | score=%s | level=%s",
                result["request_id"], score, level)
    return result


@app.get(
    "/predictions", response_model=list[PredictionResponse], tags=["Predictions"],
    summary="Список прогнозов с фильтрацией",
    responses={400: {"description": "Неверный risk_level"}},
)
def get_predictions(
    limit: int = Query(default=10, ge=1, le=100),
    risk_level: str | None = Query(default=None, description="low, medium, high"),
):
    validate_risk_filter(risk_level)
    return list_predictions(limit=limit, risk_level=risk_level)


@app.get(
    "/predictions/{request_id}", response_model=PredictionResponse,
    tags=["Predictions"], summary="Найти прогноз по request_id",
    responses={404: {"description": "Прогноз не найден"}},
)
def get_prediction_by_id(request_id: str):
    result = get_prediction(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return result


# ------------------------ Дополнительные функции -------------------------

@app.get(
    "/equipment/{equipment_id}/predictions",
    response_model=list[PredictionResponse], tags=["Additional"],
    summary="История прогнозов одного оборудования",
)
def equipment_history(equipment_id: str):
    return get_equipment_predictions(equipment_id)


@app.get(
    "/predictions/{request_id}/explain", response_model=ExplainResponse,
    tags=["Additional"], summary="Объяснить расчёт риска",
    responses={404: {"description": "Прогноз не найден"}},
)
def explain_prediction(request_id: str):
    result = get_prediction(request_id)
    source = get_source_request(request_id)
    if result is None or source is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {
        "request_id": request_id,
        "equipment_id": result["equipment_id"],
        "base_score": BASE_SCORE,
        "factors": get_risk_factors(source),
        "risk_score": result["risk_score"],
        "explanation": (
            "Оценка = min(1.0, базовый балл + веса сработавших правил). "
            "Пороги и веса являются учебными, не результатом ML-обучения."
        ),
    }


@app.delete(
    "/predictions/{request_id}", status_code=status.HTTP_204_NO_CONTENT,
    tags=["Additional"], summary="Удалить прогноз и исходные данные",
    responses={404: {"description": "Прогноз не найден"}},
)
def remove_prediction(request_id: str):
    if not delete_prediction(request_id):
        raise HTTPException(status_code=404, detail="Prediction not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
