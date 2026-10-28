"""Лабораторная работа № 2: развитие Equipment Failure Risk API."""

import logging
import math
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from batching import DynamicBatcher
from config import Settings
from llm import build_explanation_prompt, call_llm
from model import BASE_SCORE, RiskModel, get_risk_factors
from schemas import (
    BatchPredictionResponse, BatchRequest, EquipmentRequest, ErrorResponse, ExplainResponse,
    HealthResponse, LLMChatRequest, LLMChatResponse, ModelInfoResponse,
    PaginatedPredictionsResponse, PredictionResponse, PredictWithLLMResponse, QueueStatusResponse,
)
from services import (
    MODEL_NAME, MODEL_TYPE, MODEL_VERSION, build_prediction, run_inference,
    validate_equipment, validate_risk_filter,
)
from storage import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("equipment-risk")
WEEKDAYS = {name: str(i) for i, name in enumerate(
    ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
)}
ERRORS = {code: {"model": ErrorResponse, "description": description} for code, description in {
    400: "Business validation error", 404: "Prediction or page not found",
    409: "Multiple predictions for request_id", 422: "Request validation error",
    429: "Inference queue is full", 500: "Internal server error",
    502: "LLM provider error", 503: "Component unavailable", 504: "Timeout",
}.items()}


def errors(*codes):
    return {code: ERRORS[code] for code in (*codes, 500)}


async def request_id_header(x_request_id: str | None = Header(default=None, description="Необязательный ID HTTP-запроса")):
    """Показывает заголовок в Swagger; присвоение ID выполняется middleware."""


def single_result(rows: list[dict]):
    if not rows:
        raise HTTPException(404, "Prediction not found")
    if len(rows) > 1:
        raise HTTPException(409, "Multiple predictions; use /requests/{request_id}/predictions")
    return rows[0]


def safe_validation_detail(value):
    """JSON не допускает NaN/Infinity даже в описании ошибочного input."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: safe_validation_detail(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_validation_detail(v) for v in value]
    return value


def create_app(settings: Settings | None = None, *, llm_transport: httpx.AsyncBaseTransport | None = None):
    cfg = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.storage = Storage(cfg.database_path)
        await app.state.storage.init_db()
        app.state.model = RiskModel(cfg)
        app.state.batcher = DynamicBatcher(app.state.model)
        async with httpx.AsyncClient(timeout=cfg.llm_timeout_seconds, transport=llm_transport) as client:
            app.state.llm_client = client
            await app.state.batcher.start()
            try:
                yield
            finally:
                await app.state.batcher.stop()

    app = FastAPI(
        title="Equipment Failure Risk API",
        description=("ЛР2: request_id, SQLite, batching и внешняя LLM. "
                     "Учебная правиловая оценка риска; задержка инференса имитируется."),
        version="2.0.0", lifespan=lifespan, dependencies=[Depends(request_id_header)],
        openapi_tags=[
            {"name": "Monitoring"}, {"name": "Predictions"}, {"name": "Batching"},
            {"name": "LLM"}, {"name": "Additional"},
        ],
    )
    app.state.settings = cfg

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        start = time.perf_counter()
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        logger.info("Request started | request_id=%s | method=%s | path=%s",
                    request_id, request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error | request_id=%s", request_id)
            response = JSONResponse(status_code=500, content={
                "detail": "Internal server error", "request_id": request_id,
            })
        elapsed = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{elapsed:.6f}"
        logger.info("Request completed | request_id=%s | status=%s | time=%.6fs",
                    request_id, response.status_code, elapsed)
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, headers=exc.headers, content=jsonable_encoder({
            "detail": exc.detail, "request_id": getattr(request.state, "request_id", None),
        }))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content=jsonable_encoder({
            "detail": safe_validation_detail(exc.errors()), "request_id": getattr(request.state, "request_id", None),
        }))

    async def score_and_save(data: EquipmentRequest, request: Request, source: str):
        validate_equipment(data)
        request_id = request.state.request_id
        scores = await run_inference(request.app.state.model, [data], [request_id])
        result = build_prediction(data, scores[0], request_id)
        await request.app.state.storage.save_many([result], [data], source)
        return result

    @app.get("/health", response_model=HealthResponse, tags=["Monitoring"], responses=errors())
    async def health():
        return {"status": "ok"}

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"], responses=errors())
    async def model_info(request: Request):
        return {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "model_type": MODEL_TYPE,
                "status": "ready" if request.app.state.model.ready else "unavailable"}

    @app.post("/predict", response_model=PredictionResponse, tags=["Predictions"],
              responses=errors(400, 422, 503, 504), summary="Оценить риск и сохранить в SQLite")
    async def predict(data: EquipmentRequest, request: Request):
        return await score_and_save(data, request, "single")

    @app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Batching"],
              responses=errors(400, 422, 503, 504), summary="Явный пакет из 1-8 устройств")
    async def batch_predict(data: BatchRequest, request: Request):
        start = time.perf_counter()
        if len(data.items) > cfg.max_batch_size:
            raise HTTPException(422, f"Batch must contain at most {cfg.max_batch_size} items")
        for item in data.items:
            validate_equipment(item)
        request_id = request.state.request_id
        scores = await run_inference(request.app.state.model, data.items, [request_id])
        results = [build_prediction(d, s, request_id) for d, s in zip(data.items, scores)]
        await request.app.state.storage.save_many(results, data.items, "batch")
        return {"request_id": request_id, "batch_size": len(results), "items": results,
                "processing_time_ms": round((time.perf_counter() - start) * 1000, 2)}

    @app.post("/predict-queued", response_model=PredictionResponse, tags=["Batching"],
              responses=errors(400, 422, 429, 503, 504), summary="Одиночный запрос через dynamic batching")
    async def predict_queued(data: EquipmentRequest, request: Request):
        validate_equipment(data)
        score = await request.app.state.batcher.submit(data, request.state.request_id)
        result = build_prediction(data, score, request.state.request_id)
        await request.app.state.storage.save_many([result], [data], "queued")
        return result

    @app.get("/queue-status", response_model=QueueStatusResponse, tags=["Monitoring"], responses=errors())
    async def queue_status(request: Request):
        return {"queue_size": request.app.state.batcher.queue.qsize(), "queue_max_size": cfg.queue_max_size,
                "max_batch_size": cfg.max_batch_size, "max_batch_wait_ms": cfg.max_batch_wait_ms,
                "inference_timeout_seconds": cfg.inference_timeout_seconds}

    # Статический путь объявляется до /predictions/{request_id}.
    @app.get("/predictions/by-weekday", response_model=PaginatedPredictionsResponse, tags=["Predictions"],
             responses=errors(400, 404, 422, 503))
    async def by_weekday(request: Request, weekday: str = Query(..., description="monday ... sunday; UTC"),
                         page: int = Query(1, ge=1), per_page: int = Query(5, ge=1, le=100)):
        weekday = weekday.strip().lower()
        if weekday not in WEEKDAYS:
            raise HTTPException(400, "weekday must be monday, tuesday, wednesday, thursday, friday, saturday or sunday")
        rows, total = await request.app.state.storage.by_weekday(WEEKDAYS[weekday], page, per_page)
        pages = (total + per_page - 1) // per_page
        if page > max(pages, 1):
            raise HTTPException(404, "Page not found")
        def link(number):
            return str(request.url.include_query_params(weekday=weekday, page=number, per_page=per_page))
        return {"data": rows, "meta": {
            "weekday": weekday, "current_page": page, "per_page": per_page,
            "total_pages": pages, "total_items": total,
            "next_page": link(page + 1) if page < pages else None,
            "prev_page": link(page - 1) if page > 1 else None,
        }}

    @app.get("/predictions", response_model=list[PredictionResponse], tags=["Predictions"],
             responses=errors(400, 422, 503))
    async def predictions(request: Request, limit: int = Query(10, ge=1, le=100), risk_level: str | None = None):
        validate_risk_filter(risk_level)
        return await request.app.state.storage.list(limit, risk_level)

    @app.get("/predictions/{request_id}", response_model=PredictionResponse, tags=["Predictions"],
             responses=errors(404, 409, 503))
    async def prediction_by_id(request_id: str, request: Request):
        return single_result(await request.app.state.storage.find(request_id))

    @app.get("/requests/{request_id}/predictions", response_model=list[PredictionResponse], tags=["Predictions"],
             responses=errors(404, 503))
    async def request_predictions(request_id: str, request: Request):
        rows = await request.app.state.storage.find(request_id)
        if not rows:
            raise HTTPException(404, "Prediction not found")
        return rows

    @app.post("/llm/chat", response_model=LLMChatResponse, tags=["LLM"], responses=errors(422, 502, 503, 504))
    async def llm_chat(data: LLMChatRequest, request: Request):
        return await call_llm(request.app.state.llm_client, data, request.state.request_id, cfg)

    @app.post("/predict-with-llm", response_model=PredictWithLLMResponse, tags=["LLM"],
              responses=errors(400, 422, 502, 503, 504), summary="Оценить риск и получить текстовое объяснение")
    async def predict_with_llm(data: EquipmentRequest, request: Request):
        prediction = await score_and_save(data, request, "single+llm")
        result = await call_llm(
            request.app.state.llm_client,
            LLMChatRequest(prompt=build_explanation_prompt(prediction), model=cfg.llm_model),
            request.state.request_id, cfg,
        )
        return {"prediction": prediction, "llm": result}

    @app.get("/equipment/{equipment_id}/predictions", response_model=list[PredictionResponse],
             tags=["Additional"], responses=errors(503))
    async def equipment_history(equipment_id: str, request: Request):
        return await request.app.state.storage.equipment_history(equipment_id)

    @app.get("/predictions/{request_id}/explain", response_model=ExplainResponse, tags=["Additional"],
             responses=errors(404, 409, 503))
    async def explain_prediction(request_id: str, request: Request):
        saved, data = await request.app.state.storage.source_request(request_id)
        return {"request_id": request_id, "equipment_id": data.equipment_id, "base_score": BASE_SCORE,
                "factors": get_risk_factors(data), "risk_score": saved["risk_score"],
                "explanation": "Оценка = min(1.0, базовый балл + веса сработавших правил). Пороги учебные."}

    @app.delete("/predictions/{request_id}", status_code=204, tags=["Additional"], responses=errors(404, 409, 503))
    async def delete_prediction(request_id: str, request: Request):
        await request.app.state.storage.delete(request_id)
        return Response(status_code=204)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
