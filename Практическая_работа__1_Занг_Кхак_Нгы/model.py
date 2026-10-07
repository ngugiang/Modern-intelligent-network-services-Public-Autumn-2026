"""Учебная интерпретируемая модель риска (правила, не обученная ML-модель)."""

from schemas import EquipmentRequest, RiskFactor

BASE_SCORE = 0.05

# (поле, порог, вес, пояснение). Правило срабатывает при значении > порога.
RULES = (
    ("temperature_c", 85, 0.25, "Температура выше 85 °C"),
    ("vibration_mm_s", 7.0, 0.25, "Вибрация выше 7 мм/с"),
    ("operating_hours", 10000, 0.15, "Наработка более 10 000 часов"),
    ("days_since_maintenance", 180, 0.15, "Обслуживание более 180 дней назад"),
    ("previous_failures", 0, 0.15, "Были предыдущие отказы"),
    ("pressure_bar", 12, 0.10, "Давление выше 12 бар"),
)


def get_risk_factors(data: EquipmentRequest) -> list[RiskFactor]:
    """Возвращает все факторы и их фактический вклад в оценку."""
    factors = []
    for field, threshold, weight, description in RULES:
        triggered = getattr(data, field) > threshold
        factors.append(
            RiskFactor(
                factor=field,
                threshold=description,
                is_triggered=triggered,
                contribution=weight if triggered else 0.0,
            )
        )
    return factors


def calculate_risk(data: EquipmentRequest) -> float:
    """0..1: базовый риск плюс вклады правил; максимум 1.0."""
    contributions = sum(f.contribution for f in get_risk_factors(data))
    return round(min(1.0, BASE_SCORE + contributions), 2)
