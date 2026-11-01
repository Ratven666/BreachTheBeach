"""Шаг 9 пайплайна — расчёт индекса волновой экспозиции WEI и рангового индекса WER.

Использует данные, сохранённые шагом 8:
  • wave_climate_daily.geojson   — дневной ряд CWEF по всем точкам
  • wave_climate_summary.geojson — сводная статистика по точкам

Алгоритм
--------
1. Для каждой точки рассчитываются четыре частных показателя:
       mean_CWEF_Wm  — среднее CWEF [Вт/м]         (фоновая нагрузка)
       E_storm_MJm   — суммарная штормовая энергия  (экстремальная составляющая)
       K_dir         — коэффициент направленной концентрации
       CV            — коэффициент вариации σ/CWEF̄

2. По ансамблю всех точек трассы присваиваются квинтильные ранги:
       R1 — ранг по mean_CWEF_Wm   ∈ [1, 5]
       R2 — ранг по E_storm_MJm    ∈ [1, 5]
       R3 — ранг по K_dir          ∈ [1, 5]
       R4 — ранг по CV             ∈ [1, 5]

3. Итоговый ранговый индекс WER (Wave Exposure Rank):
       WER = (R1 · R2 · R3 · R4)^(1/4)            (1)
       WER ∈ [1, 5]

Результат сохраняется в:
  • wave_exposure_index.geojson  — summary + показатели WEI + ранги + WER
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.waves.index_service import WaveExposureIndexService

# ── Корень рабочей директории ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE         = PROJECT_ROOT / "nvrsk_calc"

# ── Входные файлы (выход шага 8) ─────────────────────────────────────────────
DAILY_IN   = BASE / "wave_climate_daily.geojson"
SUMMARY_IN = BASE / "wave_climate_summary.geojson"

# ── Выходной файл ────────────────────────────────────────────────────────────
INDEX_OUT  = BASE / "wave_exposure_index.geojson"

# ── Параметры расчёта ────────────────────────────────────────────────────────
# Штормовой порог: задаётся через перцентиль CWEF (Harley 2017).
# Для явного порога [Вт/м] раскомментируйте STORM_THRESHOLD.
STORM_PERCENTILE = 90.0
# STORM_THRESHOLD = 5000.0   # Вт/м — явный порог (переопределяет перцентиль)
STORM_THRESHOLD: float | None = None

# Временной шаг ряда (86 400 с = 1 сутки)
DT_SECONDS = 86_400.0


# ─────────────────────────────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("Шаг 9: расчёт WEI-показателей и рангового индекса WER")
    logger.info("=" * 60)

    for path, label in [(DAILY_IN, "Дневной ряд"), (SUMMARY_IN, "Сводная статистика")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} не найден: {path}\n"
                "Убедитесь, что шаг 8 пайплайна выполнен успешно."
            )

    svc = WaveExposureIndexService(
        storm_percentile=STORM_PERCENTILE,
        storm_threshold=STORM_THRESHOLD,
        dt_seconds=DT_SECONDS,
    )

    out_path = svc.export(
        daily_geojson_path=DAILY_IN,
        summary_geojson_path=SUMMARY_IN,
        output_path=INDEX_OUT,
    )

    logger.success(f"Шаг 9 завершён. Результат: {out_path}")
    logger.info(
        "Добавленные поля:\n"
        "  Показатели : mean_CWEF_Wm, E_storm_MJm, storm_threshold_Wm,\n"
        "               K_dir, CV, n_days, n_storm_days, top3_sectors\n"
        "  Ранги      : R1, R2, R3, R4\n"
        "  Итоговый   : WER  (геометрическое среднее R1·R2·R3·R4)"
    )


if __name__ == "__main__":
    main()
