# Equipment Failure Risk API — Лабораторная работа 2

Развитие собственного проекта ЛР1 по оценке риска отказа оборудования.
Сохранены `EquipmentRequest`, шесть правил расчёта риска и дополнительные
функции: история устройства, объяснение правил и удаление результата.

## Установка и запуск

Python 3.10 или новее. Команды выполняются из папки с `main.py`.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Linux/macOS:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn main:app --reload
```

Swagger: <http://127.0.0.1:8000/docs>

OpenAPI: <http://127.0.0.1:8000/openapi.json>

Для эксперимента с очередью используйте **один worker**. Очередь принадлежит
одному процессу. Swagger загружает свои JS/CSS с CDN; браузеру нужен доступ
к интернету для загрузки интерфейса.

## Что изменилось после ЛР1

- Middleware принимает `X-Request-ID` или генерирует UUID, сохраняет его
  в `request.state` и добавляет `X-Request-ID`, `X-Process-Time` к ответу.
- Идентификатор проходит через инференс, журнал и SQLite.
- In-memory словари заменены на SQLite и `aiosqlite` с WAL и параметрами SQL.
- `POST /predict` возвращает **200 OK после COMMIT** вместо 201 в ЛР1:
  это соответствует условию ЛР2.
- Добавлены чтение всего batch, пагинация по дню недели, explicit и dynamic
  batching, ограничение очереди, timeout и асинхронная HTTP-интеграция LLM.
- Ошибки имеют поля `detail`, `request_id`. У 429 есть `Retry-After: 1`.

В методичке пример относится к агроскорингу. Здесь предметная область ЛР1
сохранена: `farm_id` заменён на `equipment_id`, `FarmRequest` — на
`EquipmentRequest`, БД называется `equipment_risk.db`.

## Endpoints и контракт

| Метод | Endpoint | Вход | Успех | Основные ошибки |
|---|---|---|---|---|
| GET | `/health` | — | 200: status | 500 |
| GET | `/model-info` | — | 200: метаданные модели | 500 |
| POST | `/predict` | EquipmentRequest | 200: PredictionResponse | 400, 422, 503, 504 |
| POST | `/batch-predict` | BatchRequest | 200: BatchPredictionResponse | 400, 422, 503, 504 |
| POST | `/predict-queued` | EquipmentRequest | 200: PredictionResponse | 400, 422, 429, 503, 504 |
| GET | `/queue-status` | — | 200: параметры очереди | 500 |
| GET | `/predictions` | limit, risk_level | 200: список прогнозов | 400, 422, 503 |
| GET | `/predictions/by-weekday` | weekday, page, per_page | 200: data и meta | 400, 404, 422, 503 |
| GET | `/predictions/{request_id}` | path request_id | 200: один прогноз | 404, 409, 503 |
| GET | `/requests/{request_id}/predictions` | path request_id | 200: весь batch | 404, 503 |
| POST | `/llm/chat` | LLMChatRequest | 200: LLMChatResponse | 422, 502, 503, 504 |
| POST | `/predict-with-llm` | EquipmentRequest | 200: prediction и llm | 400, 422, 502, 503, 504 |
| GET | `/equipment/{equipment_id}/predictions` | equipment_id | 200: история устройства | 503 |
| GET | `/predictions/{request_id}/explain` | request_id | 200: вклады правил | 404, 409, 503 |
| DELETE | `/predictions/{request_id}` | request_id | 204 | 404, 409, 503 |

Дополнительно любой endpoint может вернуть 500 при неожиданной ошибке.
`risk_level=critical` возвращает 400, как в ЛР1.

Образцы JSON: `examples/equipment.json`, `examples/batch.json`,
`examples/llm_chat.json`. Их можно вставить в Swagger → Try it out.
`X-Request-ID` доступен как необязательный header-параметр в Swagger.

## Правила оценки

База: 0.05. Вклад начисляется при строгом превышении порога.

| Поле | Порог | Вклад |
|---|---:|---:|
| temperature_c | > 85 | 0.25 |
| vibration_mm_s | > 7 | 0.25 |
| operating_hours | > 10000 | 0.15 |
| days_since_maintenance | > 180 | 0.15 |
| previous_failures | > 0 | 0.15 |
| pressure_bar | > 12 | 0.10 |

Результат: `round(min(1.0, 0.05 + сумма вкладов), 2)`.
Категории: `<0.3` — low; `<0.7` — medium; остальные — high.
Версия правил осталась 1.0; версия API — 2.0.0.

Это учебная правиловая модель. Она не обучалась на данных.
Задержка `asyncio.sleep(overhead + per_item * batch_size)` имитирует работу
модели для экспериментов с очередью. Тест группировки не доказывает
ускорение реальной ML-модели.

## SQLite и request_id

Файл `equipment_risk.db` создаётся рядом с `config.py` при запуске.
Поля таблицы: `id`, `request_id`, `equipment_id`, `risk_score`, `risk_level`,
`recommendation`, `model_version`, `source`, `created_at`, `input_json`.
Последнее поле сохраняет исходные данные для объяснения правил после
перезапуска. Через обычные ответы API оно не выдаётся.

`request_id` — идентификатор HTTP-запроса, а `id` — уникальный ключ строки.
В explicit batch у всех результатов один `request_id`; каждый элемент
сохраняется отдельной строкой. Поэтому одиночный поиск может вернуть 409.
Повторный POST с тем же request_id создаёт новые строки: идемпотентность
не реализована. GET получает новый HTTP ID, а в теле результата остаётся
идентификатор исходного POST.

Одиночная и пакетная запись используют `executemany` в одной транзакции
с явными COMMIT/ROLLBACK. В batch один вызов модели и один COMMIT.
Пагинация использует COUNT, LIMIT, OFFSET в одной read-транзакции.

`created_at` хранится в UTC. Фильтр weekday выбирает записи за все даты
соответствующего дня недели в UTC. `page>=1`, `1<=per_page<=100`.
Пустая первая страница: 200, `data=[]`, `total_pages=0`.
Несуществующая страница: 404. Некорректный weekday: 400.

## Настройки

Параметры задаются переменными окружения перед запуском сервера.

| Переменная | По умолчанию |
|---|---|
| DATABASE_PATH | equipment_risk.db рядом с config.py |
| MAX_BATCH_SIZE | 8 (допустимо 1–8) |
| MAX_BATCH_WAIT_MS | 50 |
| QUEUE_MAX_SIZE | 100 |
| INFERENCE_TIMEOUT_SECONDS | 2.0 |
| MODEL_FIXED_OVERHEAD_SECONDS | 0.020 |
| MODEL_PER_ITEM_SECONDS | 0.003 |
| LLM_BASE_URL | https://vireonix.ai/v1 |
| LLM_MODEL | auto |
| LLM_TIMEOUT_SECONDS | 60.0 |

Worker собирает до 8 элементов или ждёт до 50 мс после первого элемента.
Каждый ожидающий HTTP-запрос получает собственный результат через Future.
В queue_size входят только элементы очереди, а не текущий batch worker.
Timeout `/predict-queued` включает ожидание в очереди и инференс. Также
отдельно ограничено время пакетного вызова модели. Запись SQLite выполняется
после инференса и не входит в его timeout; таймаут соединения SQLite — 5 с.

## LLM

Согласно методичке используется OpenAI-совместимый endpoint Vireonix
`https://vireonix.ai/v1/chat/completions`, модель `auto`, без Authorization
и API-ключа. Один `httpx.AsyncClient` создаётся в lifespan и закрывается
при остановке. Timeout LLM ограничивает и HTTP-операцию, и общее ожидание.

`/predict-with-llm` сначала рассчитывает и **сохраняет** прогноз, затем
запрашивает объяснение. LLM получает только risk_score, risk_level,
recommendation, model_version. ID оборудования и исходные измерения
в prompt не включаются. Ответ модели не может перезаписать числовой
результат, так как `prediction` и `llm` формируются независимо.

При ошибке LLM прогноз остаётся в БД: используйте request_id из ошибки
в `GET /predictions/{request_id}`.

Ошибки provider/структуры ответа → 502; соединения → 503; timeout → 504.
Внешний сервис может быть недоступен. В проведённой прямой проверке
28.10.2026 Vireonix вернул 502 с сообщением о временной недоступности.
Скриншоты успешных LLM-ответов в материалах проверки получены с **явной
локальной тестовой фикстурой `test-fixture`** и не доказывают доступность
живого провайдера. В обычном `main:app` фикстура не включена.

## Проверка

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Тесты изолированы временными БД и не отправляют запросы к внешней LLM.
Успешные и ошибочные ответы провайдера задаются через httpx.MockTransport.
Проверяются HTTP-контракт, корреляция ID, граничные данные, сохранность
после перезапуска, rollback всего batch, конкурентные запросы, 429/504,
отмена Future и остановка worker, а также сохранение prediction при сбое LLM.

Отдельный тест через реальный HTTP (сервер должен работать):

```powershell
.\.venv\Scripts\python.exe test_dynamic_batch.py
```

Он отправляет 16 запросов и проверяет коды, ID устройств, request_id и
баллы риска. Для доказательства группировки смотрите строки
`Dynamic batch completed` и `Inference completed` в журнале сервера.

### Проверка timeout в Swagger

Остановите сервер, затем в PowerShell:

```powershell
$env:INFERENCE_TIMEOUT_SECONDS = "0.001"
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

`POST /predict` должен вернуть 504 и request_id. После эксперимента
остановите сервер и восстановите значение:

```powershell
Remove-Item Env:INFERENCE_TIMEOUT_SECONDS
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

### Переполнение очереди

Остановите сервер и задайте:

```powershell
$env:QUEUE_MAX_SIZE = "1"
$env:MAX_BATCH_SIZE = "1"
$env:MODEL_FIXED_OVERHEAD_SECONDS = "0.2"
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Запустите `test_dynamic_batch.py` во втором терминале. При перегрузке часть
запросов получает 429; скрипт штатно завершится с ошибкой, поскольку его
обычный сценарий требует 16 ответов 200. Конкретное число 429 зависит от
планирования запросов. Детерминированная проверка есть в pytest.
После эксперимента удалите три переменные и перезапустите сервер.

## Структура

- main.py — приложение, lifespan, middleware, обработчики ошибок и маршруты;
- config.py — настройки окружения;
- schemas.py — схемы запросов и ответов;
- model.py — исходные правила ЛР1 и пакетный интерфейс;
- services.py — бизнес-проверки, timeout, постобработка;
- storage.py — асинхронный SQLite, транзакции, чтение и пагинация;
- batching.py — Queue, worker и Future;
- llm.py — внешний HTTP API и prompt объяснения;
- examples/ — готовые JSON-запросы;
- tests/ — автоматическая проверка;
- test_dynamic_batch.py — конкурентный тест через HTTP;
- CHECKLIST_VI.md — порядок запуска и проверки на русском задании.

Исходный репозиторий ЛР1:
<https://github.com/ngugiang/Modern-intelligent-network-services-Public-Autumn-2026>
Новые файлы ЛР2 нужно добавить отдельной папкой в свой репозиторий перед
сдачей; в ходе этой работы удалённый репозиторий не изменялся.

## Уточнения после сравнения с полным образцом

Внешний вызов явно использует stream=false. Если provider не вернул model,
returned_model равен null; некорректные типы и пустой ответ дают 502.
Если в запросе /llm/chat поле model не указано, используется LLM_MODEL
из окружения; явно переданное значение имеет приоритет.

Модель проверяется не только по длине результата: NaN, Infinity, bool,
строки и баллы вне 0–1 преобразуются в 503 без записи. Неконечные числа
во входе возвращают корректный JSON с 422 и request_id.

Скрипт test_dynamic_batch.py отправляет low/medium/high (0.05/0.55/1.0),
проверяет категории и X-Request-ID в header и body. Ошибки подключения
выводятся как CLIENT_ERROR; при любой неудаче скрипт завершается с кодом 1.

verification/pytest.txt, junit.xml, concurrency.json, server.log и
openapi.json обновлены для этой версии. Скриншоты и http_results.json
сохранены от предыдущего запуска; успешные LLM-снимки остаются фикстурами.
В Word-отчёте предыдущей версии указаны 67 тестов; актуальная версия
кода проходит 86 тестов.
