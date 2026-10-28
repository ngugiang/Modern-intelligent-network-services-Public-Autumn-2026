"""Постоянное хранилище; все SQL-операции выполняются через aiosqlite."""

import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastapi import HTTPException

from schemas import EquipmentRequest, PredictionResponse

logger = logging.getLogger("equipment-risk.storage")
COLUMNS = "request_id, equipment_id, risk_score, risk_level, recommendation, model_version, created_at"
INSERT_SQL = """
INSERT INTO predictions
(request_id, equipment_id, risk_score, risk_level, recommendation,
 model_version, source, created_at, input_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class Storage:
    def __init__(self, path: Path):
        self.path = path

    @asynccontextmanager
    async def connection(self):
        try:
            async with aiosqlite.connect(str(self.path), timeout=5.0) as db:
                db.row_factory = aiosqlite.Row
                yield db
        except (sqlite3.Error, OSError) as exc:
            logger.exception("SQLite operation failed")
            raise HTTPException(503, "Prediction storage is unavailable") from exc

    async def init_db(self):
        async with self.connection() as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    equipment_id TEXT NOT NULL,
                    risk_score REAL NOT NULL CHECK(risk_score BETWEEN 0 AND 1),
                    risk_level TEXT NOT NULL CHECK(risk_level IN ('low','medium','high')),
                    recommendation TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    input_json TEXT NOT NULL
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_request ON predictions(request_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_equipment ON predictions(equipment_id)")
            await db.commit()

    async def save_many(self, results: list[PredictionResponse], inputs: list[EquipmentRequest], source: str):
        if len(results) != len(inputs):
            raise ValueError("Predictions and inputs must have equal lengths")
        rows = [
            (p.request_id, p.equipment_id, p.risk_score, p.risk_level,
             p.recommendation, p.model_version, source, p.created_at.isoformat(), d.model_dump_json())
            for p, d in zip(results, inputs)
        ]
        async with self.connection() as db:
            await db.execute("BEGIN")
            try:
                await db.executemany(INSERT_SQL, rows)
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
        logger.info("SQLite committed | request_ids=%s | count=%s | source=%s",
                    [p.request_id for p in results], len(rows), source)

    async def find(self, request_id: str) -> list[dict]:
        async with self.connection() as db:
            async with db.execute(
                f"SELECT {COLUMNS} FROM predictions WHERE request_id=? ORDER BY id", (request_id,)
            ) as cursor:
                return [dict(row) for row in await cursor.fetchall()]

    async def list(self, limit: int, risk_level: str | None):
        sql = f"SELECT {COLUMNS} FROM predictions"
        params = []
        if risk_level is not None:
            sql += " WHERE risk_level=?"
            params.append(risk_level)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        async with self.connection() as db:
            async with db.execute(sql, params) as cursor:
                return [dict(row) for row in await cursor.fetchall()]

    async def by_weekday(self, weekday: str, page: int, per_page: int):
        async with self.connection() as db:
            # COUNT и страница читаются из одного снимка БД.
            await db.execute("BEGIN")
            async with db.execute(
                "SELECT COUNT(*) FROM predictions WHERE strftime('%w',created_at)=?", (weekday,)
            ) as cursor:
                total = (await cursor.fetchone())[0]
            async with db.execute(
                f"SELECT {COLUMNS} FROM predictions WHERE strftime('%w',created_at)=? "
                "ORDER BY id DESC LIMIT ? OFFSET ?", (weekday, per_page, (page - 1) * per_page)
            ) as cursor:
                rows = [dict(row) for row in await cursor.fetchall()]
            await db.commit()
        return rows, total

    async def equipment_history(self, equipment_id: str):
        async with self.connection() as db:
            async with db.execute(
                f"SELECT {COLUMNS} FROM predictions WHERE equipment_id=? ORDER BY id DESC",
                (equipment_id,),
            ) as cursor:
                return [dict(row) for row in await cursor.fetchall()]

    async def source_request(self, request_id: str):
        async with self.connection() as db:
            async with db.execute(
                f"SELECT {COLUMNS}, input_json FROM predictions WHERE request_id=? ORDER BY id", (request_id,)
            ) as cursor:
                rows = await cursor.fetchall()
        if not rows:
            raise HTTPException(404, "Prediction not found")
        if len(rows) > 1:
            raise HTTPException(409, "Multiple predictions; use /requests/{request_id}/predictions")
        return dict(rows[0]), EquipmentRequest.model_validate_json(rows[0]["input_json"])

    async def delete(self, request_id: str):
        async with self.connection() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT id FROM predictions WHERE request_id=?", (request_id,)
                ) as cursor:
                    rows = await cursor.fetchall()
                if not rows:
                    raise HTTPException(404, "Prediction not found")
                if len(rows) > 1:
                    raise HTTPException(409, "Multiple predictions; deletion is ambiguous")
                await db.execute("DELETE FROM predictions WHERE id=?", (rows[0]["id"],))
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
