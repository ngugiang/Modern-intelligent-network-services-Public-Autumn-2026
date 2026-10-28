"""HTTP-интеграция LLM. LLM создаёт только текст и не пересчитывает риск."""

import asyncio
import json
import logging
import time

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from config import Settings
from schemas import LLMChatRequest, LLMChatResponse, PredictionResponse

logger = logging.getLogger("equipment-risk.llm")


async def call_llm(client: httpx.AsyncClient, data: LLMChatRequest, request_id: str, settings: Settings):
    started = time.perf_counter()
    payload = {
        "model": data.model if "model" in data.model_fields_set else settings.llm_model,
        "messages": [
            {"role": "system", "content": (
                "Ты помощник учебного сервиса оценки риска отказа оборудования. "
                "Отвечай кратко на русском языке. Не изменяй risk_score и risk_level. "
                "Не придумывай причины риска или измерения. "
                "Не представляй учебную оценку как инженерное заключение."
            )},
            {"role": "user", "content": data.prompt},
        ],
        "temperature": data.temperature,
        "max_tokens": data.max_tokens,
        "stream": False,
    }
    logger.info("LLM request started | request_id=%s | model=%s", request_id, payload["model"])
    try:
        response = await asyncio.wait_for(
            client.post(
                settings.llm_base_url.rstrip("/") + "/chat/completions",
                json=payload,
                headers={"X-Request-ID": request_id},
            ),
            timeout=settings.llm_timeout_seconds,
        )
    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        logger.warning("LLM timeout | request_id=%s", request_id)
        raise HTTPException(504, "LLM timeout") from exc
    except httpx.RequestError as exc:
        logger.warning("LLM connection failed | request_id=%s | error=%s", request_id, type(exc).__name__)
        raise HTTPException(503, "LLM service is unavailable") from exc
    if response.status_code != 200:
        logger.warning("LLM provider error | request_id=%s | status=%s", request_id, response.status_code)
        raise HTTPException(502, f"LLM provider returned an error (HTTP {response.status_code})")
    try:
        body = response.json()
        choice = body["choices"][0]
        result = LLMChatResponse(
            request_id=request_id,
            requested_model=payload["model"],
            returned_model=body.get("model"),
            answer=choice["message"]["content"],
            finish_reason=choice.get("finish_reason"),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        result.answer = result.answer.strip()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError, ValidationError) as exc:
        logger.warning("Invalid LLM response | request_id=%s", request_id)
        raise HTTPException(502, "Invalid response from LLM provider") from exc
    logger.info("LLM request completed | request_id=%s | latency_ms=%s", request_id, result.latency_ms)
    return result


def build_explanation_prompt(prediction: PredictionResponse):
    # Не передаём идентификатор оборудования или исходные измерения.
    summary = prediction.model_dump(mode="json", include={
        "risk_score", "risk_level", "recommendation", "model_version"
    })
    return (
        "Кратко объясни результат учебной правиловой оценки риска отказа оборудования. "
        "Не изменяй risk_score и risk_level, не придумывай причины или измерения. "
        "Оценка не является инженерным заключением. Результат: "
        + json.dumps(summary, ensure_ascii=False)
    )
