# Equipment Failure Risk API

## Практическая работа № 1 — Современные интеллектуальные сетевые сервисы

**Тема:** «Сервис оценки риска отказа оборудования».

Учебный REST API на **Python + FastAPI + Pydantic**. По техническим
характеристикам двигателя, насоса, компрессора или генератора система
рассчитывает показатель риска от 0 до 1, категорию `low / medium / high`
и рекомендацию по техническому обслуживанию.

> Важно: демонстрационная **правиловая модель**, НЕ обученная ML-модель.
> Пороги и веса выбраны для практической работы, не являются инженерными
> стандартами безопасности и не подходят для принятия реальных решений
> об эксплуатации оборудования.

## 1. Состав проекта

```text
equipment_risk_api/
├── main.py              # маршруты, ошибки, middleware, логирование
├── schemas.py           # схемы Pydantic и документация полей
├── model.py             # расчёт риска + вклад факторов
├── services.py          # бизнес-проверки, риск-профиль, рекомендации
├── storage.py           # временное хранилище (RAM)
├── requirements.txt     # зависимости
├── README.md            # инструкция
└── tests/
    └── test_api.py      # автоматические тесты
```

## 2. Установка и запуск

Требуется Python **3.10+**.

В командной строке из папки `equipment_risk_api`:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Установить зависимости и запустить API:

```bash
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI: <http://127.0.0.1:8000/openapi.json>
- Проверка API: <http://127.0.0.1:8000/health>

## 3. Контракт API

| Метод | Endpoint | Назначение | Успех | Ошибки |
|---|---|---|---|---|
| GET | `/health` | Доступность API | 200 | 500 |
| GET | `/model-info` | Информация о модели | 200 | 500 |
| POST | `/predict` | Создать прогноз | 201 | 400, 422, 503, 500 |
| GET | `/predictions/{request_id}` | Прогноз по ID | 200 | 404, 500 |
| GET | `/predictions` | Список и фильтр `limit` (1–100), `risk_level` | 200 | 400, 422, 500 |
| GET | `/equipment/{equipment_id}/predictions` | История одного устройства (доп.) | 200 | 500 |
| GET | `/predictions/{request_id}/explain` | Причины риска (доп.) | 200 | 404, 500 |
| DELETE | `/predictions/{request_id}` | Удаление прогноза (доп.) | 204 | 404, 500 |

`X-Process-Time` добавляется в заголовки ответов.

### Пример POST /predict (высокий риск)

```json
{
  "equipment_id": "MOTOR-001",
  "equipment_type": "motor",
  "temperature_c": 92.5,
  "vibration_mm_s": 8.4,
  "operating_hours": 12000,
  "pressure_bar": 14.0,
  "days_since_maintenance": 220,
  "previous_failures": 2
}
```

Успех — `201 Created`. В ответе содержатся:
`request_id`, `equipment_id`, `risk_score`, `risk_level`,
`recommendation`, `model_version`, `created_at`.
Для примера выше `risk_score = 1.0`, `risk_level = "high"`.

### Примеры ошибок

- `422`: `"temperature_c": 250` — нарушение ограничения Pydantic.
- `400`: `"equipment_type": "robot"` — неизвестный тип оборудования.
- `404`: `GET /predictions/nonexistent-id` — прогноз не найден.
- `422`: `GET /predictions?limit=-5` — недопустимый лимит.
- `400`: `GET /predictions?risk_level=critical` — неизвестный уровень.
- `503`: при установке `MODEL_READY = False` в `main.py`.

## 4. Правила модели

Базовая оценка — `0.05`. Каждый фактор срабатывает при **строгом превышении**
порога ( `>` ):

| Фактор | Порог | Добавка |
|---|---:|---:|
| Температура | > 85 °C | +0.25 |
| Вибрация | > 7 мм/с | +0.25 |
| Наработка | > 10 000 часов | +0.15 |
| Дней после ТО | > 180 | +0.15 |
| Предыдущие отказы | > 0 | +0.15 |
| Давление | > 12 бар | +0.10 |

`risk_score = round(min(1.0, 0.05 + сумма добавок), 2)`.

- **low:** `risk_score < 0.3`;
- **medium:** `0.3 <= risk_score < 0.7`;
- **high:** `risk_score >= 0.7`.

При `GET /predictions/{request_id}/explain` выдаются все правила,
их срабатывания и индивидуальные вклады.

## 5. Параметры и валидация

| Поле | Тип | Ограничения |
|---|---|---|
| `equipment_id` | str | длина 1–50 |
| `equipment_type` | str | motor, pump, compressor, generator |
| `temperature_c` | float | −40 … 200 |
| `vibration_mm_s` | float | 0 … 100 |
| `operating_hours` | int | ≥ 0 |
| `pressure_bar` | float | 0 … 50 |
| `days_since_maintenance` | int | ≥ 0 |
| `previous_failures` | int | ≥ 0 |

Все восемь полей обязательны, лишние поля запрещены. Тип и диапазон
проверяются Pydantic (422), справочник типов — бизнес-логикой (400).

## 6. Тестирование

```bash
python -m pytest tests -v
```

Используется `fastapi.testclient.TestClient`; тесты изолированы друг от друга.
Для отчёта в Swagger UI сделайте **четыре скриншота**:

1. Страница `/docs` с перечнем endpoints.
2. Успешный `POST /predict` с кодом `201`.
3. `POST /predict` с `temperature_c = 250` — `422`.
4. `GET /predictions/unknown-id` — `404`.

Для таблицы test cases дополнительно проверьте:
`GET /health` — 200; `GET /model-info` — 200;
`POST /predict` c `equipment_type="robot"` — 400;
поиск существующего `request_id` — 200;
`GET /predictions?limit=2` — 200;
`GET /predictions?risk_level=high` — 200;
`GET /predictions?limit=-5` — 422.

## 7. Архитектура и ограничения

```text
Клиент (Swagger / CRM / UI)
           |
        HTTP/JSON
           v
       main.py (FastAPI)
       /      |       \
 schemas.py model.py services.py
       \      |       /
        storage.py (RAM)
           |
        JSON-ответ
```

Вычисление выполняется **синхронно**. Результат хранится только в
оперативной памяти одного процесса, после перезапуска исчезает.
Отсутствуют авторизация, постоянная БД и производственная модель;
для реального применения нужны контролируемые параметры по типу
оборудования, обученная/валидированная модель, надёжное хранение и
мониторинг. `GET /predictions` возвращает новые записи первыми.
