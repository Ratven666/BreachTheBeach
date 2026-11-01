from __future__ import annotations

"""Шаг 5.5 — экспорт индексов из wave_activity_indices.db в GeoJSON.

EXPORT_MODE:
    "trends"  — один объект на точку (средние, CV, тренды);
    "periods" — объект на пару точка x период (временной слой).

Все пути и параметры задаются переменными ниже.
"""

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "Kaliningrad" / "data"
INDICES_DB = DATA_DIR / "db" / "wave_activity_indices.db"
OUTPUT_GEOJSON = DATA_DIR / "wave" / "wave_activity_indices_trends.geojson"

EXPORT_MODE = "trends"
GEOJSON_INDENT: int | None = None
LOG_LEVEL = "INFO"


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL)

    from src.waves.activity_indices import export_geojson

    stats = export_geojson(
        INDICES_DB, OUTPUT_GEOJSON, mode=EXPORT_MODE, indent=GEOJSON_INDENT
    )
    logger.success("{} features -> {}", stats.feature_count, stats.output_path)


if __name__ == "__main__":
    main()
