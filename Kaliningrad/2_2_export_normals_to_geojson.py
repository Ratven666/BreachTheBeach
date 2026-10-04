from __future__ import annotations

"""
pypeline/3_1_export_normals_to_geojson.py

Экспортирует нормали береговой линии из SQLite-базы в GeoJSON-файл.
Геометрия — LineString от исходной точки до конца вектора нормали (EPSG:4326).

Запуск:
    python pypeline/3_1_export_normals_to_geojson.py
"""

import os
from pathlib import Path

from loguru import logger

# --- Конфигурация --------------------------------------------------------

DATABASE_PATH   = Path("data/db/coastline.db")
OUTPUT_PATH     = Path("data/coastline/exported_normals.geojson")
SOURCE_NAME     = "kaliningrad_coastline_step200m_right"
SEA_SIDE: str | None = None   # напр. "right" или "left"
SOURCE_ID: int | None = None   # если задан — игнорирует SOURCE_NAME/SEA_SIDE
NORMAL_LENGTH_M: float = 300.0

# -------------------------------------------------------------------------


def setup_env() -> None:
    if "COASTLINE_DATABASE_URL" not in os.environ:
        url = f"sqlite:///{DATABASE_PATH.resolve().as_posix()}"
        os.environ["COASTLINE_DATABASE_URL"] = url
        logger.debug(f"COASTLINE_DATABASE_URL set to: {url}")


def main() -> None:
    setup_env()

    from src.coastline.exporters.SQLiteNormalsToGeoJsonExporter import (
        SQLiteNormalsToGeoJsonExporter,
    )

    exporter = SQLiteNormalsToGeoJsonExporter()

    sources = exporter.list_sources()
    if not sources:
        logger.warning(
            "No normal sources found in the database. "
            "Run normals calculation pipeline first."
        )
        return

    logger.info(f"Normal sources in DB ({len(sources)}):")
    for s in sources:
        logger.info(
            f"  id={s['id']}  name={s['name']!r}  "
            f"sea_side={s['sea_side']}  normals={s['normals_count']}  "
            f"working_crs={s['working_crs']}  created={s['created_at']}"
        )

    if SOURCE_ID is not None:
        result = exporter.export_by_id(
            normal_source_id=SOURCE_ID,
            output_path=OUTPUT_PATH,
            normal_length_m=NORMAL_LENGTH_M,
        )
    else:
        result = exporter.export(
            source_name=SOURCE_NAME,
            output_path=OUTPUT_PATH,
            normal_length_m=NORMAL_LENGTH_M,
            sea_side=SEA_SIDE,
        )

    logger.success(f"Done. GeoJSON saved to: {result.resolve()}")


if __name__ == "__main__":
    main()
