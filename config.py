"""Настройки прототипа; переменные окружения читаются при старте приложения."""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH_SIZE = 8
PROJECT_DIR = Path(__file__).resolve().parent


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    database_path: Path = PROJECT_DIR / "equipment_risk.db"
    max_batch_size: int = Field(default=8, ge=1, le=MAX_BATCH_SIZE)
    max_batch_wait_ms: float = Field(default=50, ge=0)
    queue_max_size: int = Field(default=100, ge=1)
    inference_timeout_seconds: float = Field(default=2.0, gt=0)
    model_fixed_overhead_seconds: float = Field(default=0.020, ge=0)
    model_per_item_seconds: float = Field(default=0.003, ge=0)
    llm_base_url: str = "https://vireonix.ai/v1"
    llm_model: str = "auto"
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    @classmethod
    def from_env(cls):
        values = {}
        for field in cls.model_fields:
            key = field.upper()
            if key in os.environ:
                values[field] = os.environ[key]
        return cls(**values)
