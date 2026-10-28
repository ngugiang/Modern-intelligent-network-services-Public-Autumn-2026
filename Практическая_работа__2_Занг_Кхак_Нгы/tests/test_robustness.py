"""Regressions after comparing the complete reference implementation."""
import json
import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from config import Settings
from main import create_app


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity'])
def test_nonfinite_input_returns_json_422(client, data, value):
    raw = json.dumps(data).replace('92.5', value)
    response = client.post('/predict', content=raw, headers={'Content-Type': 'application/json'})
    assert response.status_code == 422
    assert response.json()['request_id'] == response.headers['X-Request-ID']
    assert client.get('/predictions').json() == []


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1, True, '0.5'])
def test_invalid_model_scores_are_503(client, app, data, monkeypatch, value):
    async def invalid(items):
        return [value] * len(items)
    monkeypatch.setattr(app.state.model, 'model_predict_batch', invalid)
    for path, body in [('/predict', data), ('/batch-predict', {'items': [data]}), ('/predict-queued', data)]:
        assert client.post(path, json=body).status_code == 503
    assert client.get('/predictions').json() == []


@pytest.mark.parametrize('returned_model', [None, 'provider-model'])
def test_llm_optional_model_and_configured_default(settings, returned_model):
    def handler(request):
        payload = json.loads(request.content)
        assert payload['stream'] is False and payload['model'] == 'configured-model'
        assert 'risk_score' in payload['messages'][0]['content']
        body = {'choices': [{'message': {'content': '  текст  '}, 'finish_reason': None}]}
        if returned_model is not None: body['model'] = returned_model
        return httpx.Response(200, json=body)
    app = create_app(settings.model_copy(update={'llm_model': 'configured-model'}),
                     llm_transport=httpx.MockTransport(handler))
    with TestClient(app) as client:
        response = client.post('/llm/chat', json={'prompt': 'test'})
        assert response.status_code == 200
        assert response.json()['requested_model'] == 'configured-model'
        assert response.json()['returned_model'] == returned_model
        assert response.json()['answer'] == 'текст'


@pytest.mark.parametrize('body', [[], None, {'choices': [None]},
    {'model': 'x', 'choices': [{'message': {'content': '   '}}]}])
def test_malformed_llm_response(settings, body):
    app = create_app(settings, llm_transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    with TestClient(app) as client:
        assert client.post('/llm/chat', json={'prompt': 'test'}).status_code == 502


def test_provider_unexpected_success_status_is_502(settings):
    app = create_app(settings, llm_transport=httpx.MockTransport(lambda request: httpx.Response(201, json={
        'model': 'x', 'choices': [{'message': {'content': 'text'}}]})))
    with TestClient(app) as client:
        response = client.post('/llm/chat', json={'prompt': 'test'})
        assert response.status_code == 502 and '201' in response.json()['detail']


@pytest.mark.parametrize('field', ['max_batch_wait_ms', 'inference_timeout_seconds', 'llm_timeout_seconds'])
def test_nonfinite_settings_rejected(field):
    with pytest.raises(ValidationError): Settings(**{field: float('inf')})
