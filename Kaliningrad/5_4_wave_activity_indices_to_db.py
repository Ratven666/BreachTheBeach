from __future__ import annotations

"""Шаг 5.4 — отдельная БД обобщающих индексов волновой активности.

Источник: wave_activity.db (дни — целые смещения от 1940-01-01 = 0).
Результат: wave_activity_indices.db с периодами и динамикой.

Все пути и параметры задаются переменными ниже.
"""

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "Kaliningrad" / "data"
ACTIVITY_DB = DATA_DIR / "db" / "wave_activity.db"
INDICES_DB = DATA_DIR / "db" / "wave_activity_indices.db"

INTERVAL_YEARS = 5
START_DAY = 0
END_DAY: int | None = None
MAX_ANGLE_DEG = 85.0
TREND_ONLY_COMPLETE_PERIODS = True
LOG_LEVEL = "INFO"


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL)

    from src.waves.activity_indices import ActivityIndexBuilder

    stats = ActivityIndexBuilder(
        ACTIVITY_DB,
        INDICES_DB,
        interval_years=INTERVAL_YEARS,
        start_day=START_DAY,
        end_day=END_DAY,
        max_angle_deg=MAX_ANGLE_DEG,
        trend_only_complete_periods=TREND_ONLY_COMPLETE_PERIODS,
    ).run()
    logger.success(
        "points={}, periods={}, days={}..{}",
        stats.n_points, stats.n_periods, stats.start_day, stats.end_day,
    )


if __name__ == "__main__":
    main()
