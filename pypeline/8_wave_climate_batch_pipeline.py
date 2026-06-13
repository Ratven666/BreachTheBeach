"""Шаг 8 пайплайна — расчёт волнового воздействия на берег.

Запускается после:
    1–3: подготовка береговой линии, точек, нормалей
    4:   метеоданные (wide-GeoJSON) → points_with_weather.geojson
    5–6: слияние линий и трассировка fetch
    7:   батиметрия (опционально)
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.waves.services.wave_climate_batch import WaveClimateBatchProcessor

BASE = Path("../nvrsk_calc")

# ── Входные файлы ────────────────────────────────────────────────────────────
POINTS_PATH   = BASE / "nvrsk_equal_radius_1000m_points_with_normals.geojson"
FETCH_CSV     = BASE / "fetch" / "fetch_by_point.csv"
# Имя совпадает с выходом шага 4
WEATHER_GEOJSON = BASE / "points_with_weather.geojson"

# ── Выходные файлы ───────────────────────────────────────────────────────────
DAILY_OUT   = BASE / "wave_climate_daily.geojson"
SUMMARY_OUT = BASE / "wave_climate_summary.geojson"

# ── Промежуточный weather CSV (in-memory — файл не записывается) ─────────────
WEATHER_CSV = BASE / "weather_expanded.csv"   # используется только при skip_expand=False


def main() -> None:
    for path in (POINTS_PATH, FETCH_CSV, WEATHER_GEOJSON):
        if not path.exists():
            raise FileNotFoundError(
                f"Входной файл не найден: {path}\n"
                "Убедитесь, что предыдущие шаги пайплайна выполнены."
            )

    processor = WaveClimateBatchProcessor()

    logger.info("Запуск волнового пайплайна (шаг 8)...")

    daily_path, summary_path = processor.export(
        points_path=POINTS_PATH,
        fetch_csv_path=FETCH_CSV,
        weather_geojson_path=WEATHER_GEOJSON,
        weather_csv_path=WEATHER_CSV,
        daily_output_path=DAILY_OUT,
        summary_output_path=SUMMARY_OUT,
    )

    logger.success(f"Дневной ряд:  {daily_path}")
    logger.success(f"Сводная карта: {summary_path}")


if __name__ == "__main__":
    main()
