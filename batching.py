"""Очередь одного процесса и worker с формированием batch по размеру/времени."""

import asyncio
import logging
from dataclasses import dataclass

from fastapi import HTTPException

from model import RiskModel
from schemas import EquipmentRequest
from services import run_inference

logger = logging.getLogger("equipment-risk.batching")


@dataclass
class WorkItem:
    data: EquipmentRequest
    request_id: str
    future: asyncio.Future


class DynamicBatcher:
    def __init__(self, model: RiskModel):
        self.model = model
        self.settings = model.settings
        self.queue = asyncio.Queue(maxsize=self.settings.queue_max_size)
        self.worker = None
        self.accepting = False

    async def start(self):
        self.accepting = True
        self.worker = asyncio.create_task(self._run(), name="equipment-batch-worker")

    async def stop(self):
        self.accepting = False
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        while not self.queue.empty():
            item = self.queue.get_nowait()
            self._fail([item], HTTPException(503, "Batch worker is stopping"))
            self.queue.task_done()

    async def submit(self, data: EquipmentRequest, request_id: str):
        if not self.accepting or self.worker is None or self.worker.done():
            raise HTTPException(503, "Batch worker is unavailable")
        if not self.model.ready:
            raise HTTPException(503, "Model is temporarily unavailable")
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait(WorkItem(data, request_id, future))
        except asyncio.QueueFull as exc:
            raise HTTPException(429, "Inference queue is full", headers={"Retry-After": "1"}) from exc
        logger.info("Queued | request_id=%s | queue_size=%s", request_id, self.queue.qsize())
        try:
            # Включает ожидание в очереди, окно формирования batch и инференс.
            return await asyncio.wait_for(future, timeout=self.settings.inference_timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise HTTPException(504, "Queued inference timeout") from exc

    @staticmethod
    def _fail(items, error):
        for item in items:
            if not item.future.done():
                item.future.set_exception(error)

    async def _run(self):
        while True:
            batch = [await self.queue.get()]
            try:
                if batch[0].future.done():
                    continue
                deadline = asyncio.get_running_loop().time() + self.settings.max_batch_wait_ms / 1000
                while len(batch) < self.settings.max_batch_size:
                    if asyncio.get_running_loop().time() >= deadline:
                        break
                    try:
                        # Уже накопленные элементы забираются без дополнительного ожидания.
                        batch.append(self.queue.get_nowait())
                        continue
                    except asyncio.QueueEmpty:
                        pass
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                    except asyncio.TimeoutError:
                        break
                active = [item for item in batch if not item.future.done()]
                if active:
                    logger.info("Dynamic batch formed | size=%s | queue_size=%s | request_ids=%s",
                                len(active), self.queue.qsize(), [item.request_id for item in active])
                    scores = await run_inference(
                        self.model, [item.data for item in active], [item.request_id for item in active]
                    )
                    for item, score in zip(active, scores):
                        if not item.future.done():
                            item.future.set_result(score)
                    logger.info("Dynamic batch completed | size=%s | request_ids=%s",
                                len(active), [item.request_id for item in active])
            except asyncio.CancelledError:
                self._fail(batch, HTTPException(503, "Batch worker is stopping"))
                raise
            except Exception as exc:
                error = exc if isinstance(exc, HTTPException) else HTTPException(503, "Batch worker failed")
                self._fail(batch, error)
                logger.exception("Dynamic batch failed | request_ids=%s", [i.request_id for i in batch])
            finally:
                for _ in batch:
                    self.queue.task_done()
