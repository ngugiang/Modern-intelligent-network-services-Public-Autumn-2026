"""16 concurrent HTTP requests: independent low/medium/high expectations."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import time
from uuid import uuid4
import httpx


def sample(base, index):
    if index % 3 == 0:
        return {**base, 'equipment_id': f'LOAD-{index:02}'}, 1.0, 'high'
    data = {**base, 'equipment_id': f'LOAD-{index:02}', 'temperature_c': 50,
            'vibration_mm_s': 2, 'operating_hours': 100, 'pressure_bar': 2,
            'days_since_maintenance': 10, 'previous_failures': 0}
    if index % 3 == 1:
        return data, 0.05, 'low'
    return {**data, 'temperature_c': 90, 'vibration_mm_s': 8}, 0.55, 'medium'


async def run(base_url):
    data = json.loads((Path(__file__).parent / 'examples/equipment.json').read_text(encoding='utf-8'))
    async with httpx.AsyncClient(base_url=base_url, timeout=15, trust_env=False) as client:
        async def send(index):
            payload, score, level = sample(data, index)
            rid = f'concurrency-{uuid4()}'
            started = time.perf_counter()
            try:
                response = await client.post('/predict-queued', json=payload, headers={'X-Request-ID': rid})
                body = response.json()
                passed = (response.status_code == 200 and isinstance(body, dict)
                          and body.get('request_id') == rid
                          and response.headers.get('X-Request-ID') == rid
                          and body.get('equipment_id') == payload['equipment_id']
                          and body.get('risk_score') == score and body.get('risk_level') == level)
                return {'index': index, 'request_id': rid, 'status': response.status_code,
                        'latency_ms': round((time.perf_counter() - started) * 1000, 2),
                        'expected_score': score, 'expected_level': level, 'passed': passed,
                        **({} if passed else {'body': body})}
            except (httpx.RequestError, ValueError) as exc:
                return {'index': index, 'request_id': rid, 'status': 'CLIENT_ERROR',
                        'passed': False, 'error': str(exc)}
        started = time.perf_counter()
        results = await asyncio.gather(*(send(i) for i in range(16)))
        elapsed = time.perf_counter() - started
        try:
            response = await client.get('/queue-status')
            queue_status = {'status': response.status_code, 'body': response.json()}
        except (httpx.RequestError, ValueError) as exc:
            queue_status = {'status': 'CLIENT_ERROR', 'error': str(exc)}
        return {'total': 16, 'passed': sum(r['passed'] for r in results),
                'http_counts': dict(Counter(str(r['status']) for r in results)),
                'seconds': round(elapsed, 3), 'requests_per_second': round(16 / elapsed, 2),
                'queue_status': queue_status, 'results': results}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_url', nargs='?', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    summary = await run(args.base_url)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary['passed'] != summary['total'] or summary['queue_status']['status'] != 200:
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
