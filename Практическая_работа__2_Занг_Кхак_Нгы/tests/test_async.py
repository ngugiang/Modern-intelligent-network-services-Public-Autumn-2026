import asyncio

import httpx
import pytest
from fastapi import HTTPException

from main import create_app
from schemas import EquipmentRequest
from services import build_prediction
from storage import Storage
from conftest import llm_ok


async def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


async def test_dynamic_batching_and_future_mapping(settings, data, low_data):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        original = app.state.model.model_predict_batch
        sizes = []
        async def observed(items):
            sizes.append(len(items))
            return await original(items)
        app.state.model.model_predict_batch = observed
        async with await client_for(app) as c:
            inputs = [{**(data if i % 2 == 0 else low_data), "equipment_id": f"D-{i}"} for i in range(16)]
            responses = await asyncio.gather(*[
                c.post("/predict-queued", json=d, headers={"X-Request-ID": f"queued-{i}"})
                for i, d in enumerate(inputs)
            ])
            assert all(r.status_code == 200 for r in responses)
            for i, r in enumerate(responses):
                assert r.json()["equipment_id"] == f"D-{i}"
                assert r.json()["request_id"] == f"queued-{i}"
                assert r.json()["risk_score"] == (1.0 if i % 2 == 0 else 0.05)
            assert sum(sizes) == 16 and max(sizes) <= 8 and max(sizes) > 1
            assert len((await c.get("/predictions?limit=100")).json()) == 16
            assert (await c.get("/queue-status")).json()["queue_size"] == 0


async def test_explicit_batch_is_one_model_call(settings, data):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        calls = []
        original = app.state.model.model_predict_batch
        async def observed(items):
            calls.append(len(items))
            return await original(items)
        app.state.model.model_predict_batch = observed
        async with await client_for(app) as c:
            assert (await c.post("/batch-predict", json={"items": [data] * 8})).status_code == 200
            assert calls == [8]


async def test_backpressure_returns_429(settings, data):
    cfg = settings.model_copy(update={"queue_max_size": 1, "max_batch_size": 1})
    app = create_app(cfg)
    async with app.router.lifespan_context(app):
        entered, release = asyncio.Event(), asyncio.Event()
        async def held(items):
            entered.set()
            await release.wait()
            return [1.0] * len(items)
        app.state.model.model_predict_batch = held
        async with await client_for(app) as c:
            first = asyncio.create_task(c.post("/predict-queued", json=data))
            await asyncio.wait_for(entered.wait(), 1)
            second = asyncio.create_task(c.post("/predict-queued", json=data))
            async def queued():
                while app.state.batcher.queue.qsize() < 1:
                    await asyncio.sleep(0)
            await asyncio.wait_for(queued(), 1)
            rejected = await c.post("/predict-queued", json=data)
            assert rejected.status_code == 429 and rejected.headers["Retry-After"] == "1"
            assert rejected.json()["request_id"] == rejected.headers["X-Request-ID"]
            release.set()
            assert [r.status_code for r in await asyncio.gather(first, second)] == [200, 200]


@pytest.mark.parametrize("path", ["/predict", "/batch-predict", "/predict-queued"])
async def test_inference_timeout_0001(settings, data, path):
    cfg = settings.model_copy(update={"inference_timeout_seconds": 0.001})
    app = create_app(cfg)
    async with app.router.lifespan_context(app):
        async with await client_for(app) as c:
            r = await c.post(path, json={"items": [data]} if path == "/batch-predict" else data)
            assert r.status_code == 504
            assert (await c.get("/predictions")).json() == []


@pytest.mark.parametrize("bad", ["raises", "length"])
async def test_model_failure_maps_to_503(settings, data, bad):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async def broken(items):
            if bad == "raises":
                raise RuntimeError("model failed")
            return []
        app.state.model.model_predict_batch = broken
        async with await client_for(app) as c:
            assert (await c.post("/predict", json=data)).status_code == 503


async def test_cancelled_submit_and_shutdown_do_not_leave_work(settings, data):
    app = create_app(settings.model_copy(update={"max_batch_wait_ms": 1000}))
    async with app.router.lifespan_context(app):
        batcher = app.state.batcher
        pending = asyncio.create_task(batcher.submit(EquipmentRequest(**data), "cancelled"))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        another = asyncio.create_task(batcher.submit(EquipmentRequest(**data), "shutdown"))
        await asyncio.sleep(0)
        await batcher.stop()
        with pytest.raises(HTTPException) as exc:
            await another
        assert exc.value.status_code == 503
        assert batcher.queue.qsize() == 0
        await asyncio.wait_for(batcher.queue.join(), 1)


async def test_batch_transaction_rolls_back(settings, data):
    storage = Storage(settings.database_path)
    await storage.init_db()
    d = EquipmentRequest(**data)
    good = build_prediction(d, 0.5, "rollback")
    invalid = good.model_copy(update={"risk_score": 2.0})
    with pytest.raises(HTTPException) as exc:
        await storage.save_many([good, invalid], [d, d], "batch")
    assert exc.value.status_code == 503
    assert await storage.find("rollback") == []


async def test_llm_request_and_explanation_score_unchanged(settings, data):
    sent = []
    def transport(request):
        import json
        sent.append(json.loads(request.content))
        assert request.headers["X-Request-ID"] == "llm-trace"
        assert "authorization" not in request.headers
        return llm_ok(request)
    app = create_app(settings, llm_transport=httpx.MockTransport(transport))
    async with app.router.lifespan_context(app):
        async with await client_for(app) as c:
            r = await c.post("/predict-with-llm", json=data, headers={"X-Request-ID": "llm-trace"})
            assert r.status_code == 200
            body = r.json()
            assert body["prediction"]["risk_score"] == 1 and body["prediction"]["risk_level"] == "high"
            assert body["llm"]["returned_model"] == "test-model"
            assert (await c.get("/predictions/llm-trace")).json() == body["prediction"]
            assert data["equipment_id"] not in sent[0]["messages"][1]["content"]


@pytest.mark.parametrize("mode,expected", [
    ("provider", 502), ("malformed", 502), ("empty", 502), ("invalid_model", 502),
    ("connection", 503), ("timeout", 504),
])
async def test_llm_errors_and_saved_prediction(settings, data, mode, expected):
    def handler(request):
        if mode == "provider": return httpx.Response(500)
        if mode == "malformed": return httpx.Response(200, text="not JSON")
        if mode == "empty": return httpx.Response(200, json={"model": "test", "choices": []})
        if mode == "invalid_model":
            return httpx.Response(200, json={"model": 123, "choices": [{"message": {"content": "x"}}]})
        if mode == "connection": raise httpx.ConnectError("offline", request=request)
        raise httpx.ReadTimeout("timeout", request=request)
    app = create_app(settings, llm_transport=httpx.MockTransport(handler))
    async with app.router.lifespan_context(app):
        async with await client_for(app) as c:
            direct = await c.post("/llm/chat", json={"prompt": "test"})
            assert direct.status_code == expected
            r = await c.post("/predict-with-llm", json=data)
            assert r.status_code == expected
            rid = r.json()["request_id"]
            assert (await c.get(f"/predictions/{rid}")).status_code == 200


async def test_llm_total_timeout(settings):
    async def delayed(request):
        await asyncio.sleep(0.2)
        return llm_ok(request)
    app = create_app(settings.model_copy(update={"llm_timeout_seconds": 0.001}),
                     llm_transport=httpx.MockTransport(delayed))
    async with app.router.lifespan_context(app):
        async with await client_for(app) as c:
            assert (await c.post("/llm/chat", json={"prompt": "test"})).status_code == 504
