from __future__ import annotations

"""Шаг 6.1: статистика волновой активности по береговым точкам -> wave_coast_stats.db.

Вход: wave_activity.db (шаг 5.1). Входная БД открывается только на чтение.
По умолчанию считаются все точки; POINTS_GEOJSON ограничивает расчёт точками из файла.
"""

import sys
from pathlib import Path

from loguru import logger


def find_project_root() -> Path:
    directory = Path(__file__).resolve().parent
    return next(p for p in (directory, *directory.parents) if (p / "src/waves/coast_stats").is_dir())


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from src.waves.coast_stats import StatsConfig, build_coast_stats  # noqa: E402

# ==================== НАСТРОЙКИ ====================

DATA_DIR = PROJECT_ROOT / "big_scale" / "data"
ACTIVITY_DB = DATA_DIR / "db/wave_activity.db"
STATS_DB = DATA_DIR / "db/wave_coast_stats.db"

# None — все точки. Иначе GeoJSON с точками: берутся point_id / id из свойств,
# а если их нет — ближайшие точки БД по координатам (в пределах MATCH_TOLERANCE_M).
POINTS_GEOJSON: Path | None = None
MATCH_TOLERANCE_M = 50.0

# Участки побережья: GeoJSON с полигонами и полем названия либо регулярная сетка (долгота, широта).
SECTIONS_GEOJSON: Path | None = None
SECTION_NAME_FIELD = "name"
SECTION_GRID_DEG = (10.0, 5.0)

YEAR_FROM: int | None = None        # None — с первого полного года расчёта волн
YEAR_TO: int | None = None
EPOCHS = None                       # ((1961, 1990), (1991, 2020)); None — две половины ряда

STORM_PERCENTILE = 90.0             # как в шаге 5.2: перцентиль по всем дням точки
ALPHA = 0.05                        # значимость после поправки FDR
MIN_PCT_PER_DECADE = 1.0            # меньше — «без изменений» даже при значимом тесте
STRONG_PCT_PER_DECADE = 5.0
PREWHITEN = True

CHUNK_POINTS = 200
N_WORKERS = 4
SAVE_POINT_ANNUAL = False           # True — таблица point_annual (годовые ряды каждой точки)

if __name__ == "__main__":
    summary = build_coast_stats(StatsConfig(
        activity_db=ACTIVITY_DB, output_db=STATS_DB,
        points_geojson=POINTS_GEOJSON, match_tolerance_m=MATCH_TOLERANCE_M,
        sections_geojson=SECTIONS_GEOJSON, section_name_field=SECTION_NAME_FIELD,
        section_grid_deg=SECTION_GRID_DEG, year_from=YEAR_FROM, year_to=YEAR_TO, epochs=EPOCHS,
        storm_percentile=STORM_PERCENTILE, alpha=ALPHA, min_pct_per_decade=MIN_PCT_PER_DECADE,
        strong_pct_per_decade=STRONG_PCT_PER_DECADE, prewhiten=PREWHITEN,
        chunk_points=CHUNK_POINTS, n_workers=N_WORKERS, save_point_annual=SAVE_POINT_ANNUAL,
    ))
    logger.info("Итог: {}", summary)
