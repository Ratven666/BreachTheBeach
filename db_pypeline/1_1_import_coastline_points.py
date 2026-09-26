# db_pypeline/1_1_import_coastline_points.py
"""
Импорт точек береговой линии из GeoJSON в БД
с фиксированным шагом по пикетажу.
"""

from __future__ import annotations

import os
from pathlib import Path

# Задаём путь к БД ДО первого импорта src.coastline.storage.db,
# чтобы DATABASE_URL вычислился с нужным значением.
_DB_PATH = Path(__file__).parent / "data" / "db" / "coastline.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("COASTLINE_DATABASE_URL", f"sqlite:///{_DB_PATH.as_posix()}")

from src.coastline.domain.CoastlineDataset import CoastlineDataset
from src.coastline.exporters.SQLitePointExporter import SQLitePointExporter
from src.coastline.point_strategies.EqualStepAlongLineStrategy import (
    EqualStepAlongLineStrategy,
)
from src.coastline.point_strategies.PointExtractionStrategy import PointSource
from src.coastline.services import CoastlinePointExtractor

# ---------------------------------------------------------------------------
GEOJSON_PATH: Path = Path("data/coastline/nvrsk_main_coastline.geojson")

STEP_M: float = 200.0
INCLUDE_ENDPOINTS: bool = True
SOURCE: PointSource = PointSource.ALL_LINES
INPUT_CRS: str | None = None
WORKING_CRS: str | None = None
DATASET_NAME: str | None = None

POINT_STRATEGY = EqualStepAlongLineStrategy(
    step_m=STEP_M,
    source=SOURCE,
    include_endpoints=INCLUDE_ENDPOINTS,
    input_crs=INPUT_CRS,
    working_crs=WORKING_CRS,
)
# ---------------------------------------------------------------------------


def main() -> None:
    dataset = CoastlineDataset.from_geojson(
        main_path=GEOJSON_PATH,
        name=DATASET_NAME or GEOJSON_PATH.stem,
    )

    extractor = CoastlinePointExtractor()
    point_set = extractor.extract(
        dataset=dataset,
        strategy=POINT_STRATEGY,
        name=DATASET_NAME or GEOJSON_PATH.stem,
    )
    point_set.print_summary()

    db_path = point_set.export(SQLitePointExporter(), output_path="")
    print(f"Сохранено → {db_path}")


if __name__ == "__main__":
    main()