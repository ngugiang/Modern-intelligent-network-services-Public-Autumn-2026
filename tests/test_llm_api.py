import pytest


def test_direct_llm_mock(client):
    r = client.post("/llm/chat", json={"prompt": "Что такое request_id?", "model": "auto",
                                      "temperature": 0.3, "max_tokens": 150})
    assert r.status_code == 200
    assert set(r.json()) == {"request_id", "requested_model", "returned_model", "answer", "finish_reason", "latency_ms"}
    assert r.json()["returned_model"] == "test-model"
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


@pytest.mark.parametrize("payload", [
    {}, {"prompt": ""}, {"prompt": "   "}, {"prompt": "x", "temperature": 3},
    {"prompt": "x", "max_tokens": 0}, {"prompt": "x", "extra": 1},
])
def test_llm_validation(client, payload):
    assert client.post("/llm/chat", json=payload).status_code == 422
