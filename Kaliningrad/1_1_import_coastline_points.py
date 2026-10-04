from __future__ import annotations

from coastline.point_strategies import EqualRadiusStrategy
from src.coastline.storage.models import Base, CoastlineSourceModel, CoastlinePointModel

"""
db_pypeline/1_1_import_coastline_points.py

Запуск (из корня проекта):
    python db_pypeline/1_1_import_coastline_points.py
"""

import json
import os
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

# ── Конфигурация ──────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = _SCRIPT_DIR / "data" / "db" / "coastline.db"

COASTLINE_MAIN_PATH  = Path("data/coastline/KaliningradOSM_all_lines.geojson")
COASTLINE_OTHER_PATH: Path | None = None
# COASTLINE_OTHER_PATH = Path("data/coastline/KaliningradOSM_other_lines.geojson")

DATASET_NAME    = "KaliningradOSM"
STRATEGY_STEP_M = 200.0
# STRATEGY_STEP_M = 1000.0
INPUT_CRS       = "EPSG:4326"
# ─────────────────────────────────────────────────────────────────────


def setup_env() -> None:
    if "COASTLINE_DATABASE_URL" not in os.environ:
        url = f"sqlite:///{DATABASE_PATH.resolve().as_posix()}"
        os.environ["COASTLINE_DATABASE_URL"] = url
        logger.debug(f"COASTLINE_DATABASE_URL → {url}")


def main() -> None:
    setup_env()

    from sqlalchemy import insert

    from src.coastline.domain.CoastlineDataset import CoastlineDataset
    from src.coastline.point_strategies import EqualStepAlongLineStrategy
    from src.coastline.point_strategies.PointExtractionStrategy import PointSource
    from src.coastline.services import CoastlinePointExtractor
    from src.coastline.storage import db as _db

    # 1. Инициализация схемы
    Base.metadata.create_all(_db.engine)
    logger.info("DB schema ready")

    # 2. Датасет береговой линии
    dataset = CoastlineDataset.from_geojson(
        main_path=str(COASTLINE_MAIN_PATH),
        other_path=str(COASTLINE_OTHER_PATH) if COASTLINE_OTHER_PATH else None,
        name=DATASET_NAME,
    )
    logger.info(
        f"Dataset: main={len(dataset.main_gdf)} feat, "
        f"other={len(dataset.other_gdf)} feat, crs={dataset.crs}"
    )

    # 3. Стратегия и извлечение точек
    strategy = EqualStepAlongLineStrategy(
        step_m=STRATEGY_STEP_M,
        source=PointSource.ALL_LINES,
        include_endpoints=True,
        working_crs=None,
        input_crs=INPUT_CRS,
    )
    strategy_params = json.dumps(
        {
            "step_m":            STRATEGY_STEP_M,
            "source":            PointSource.MAIN_ONLY.value,
            "include_endpoints": True,
            "working_crs":       None,
            "input_crs":         INPUT_CRS,
        },
        ensure_ascii=False,
    )

    extractor = CoastlinePointExtractor()
    point_set = extractor.extract(dataset=dataset, strategy=strategy, name=DATASET_NAME)
    point_set.print_summary()

    # 4. Запись в БД
    with _db.SessionLocal() as session:

        source = CoastlineSourceModel(
            name=point_set.meta.name,
            strategy_name=point_set.meta.strategy_name,
            source_mode=point_set.meta.source_mode,
            strategy_params=strategy_params,
            main_geojson_path=str(COASTLINE_MAIN_PATH.resolve()),
            other_geojson_path=(
                str(COASTLINE_OTHER_PATH.resolve())
                if COASTLINE_OTHER_PATH else None
            ),
            crs=str(point_set.gdf.crs) if point_set.gdf.crs else None,
            points_count=len(point_set.gdf),
        )
        session.add(source)
        session.flush()

        rows_data = [
            {
                "source_id": source.id,
                "seq":       int(seq),
                "lon":       float(row.geometry.x),
                "lat":       float(row.geometry.y),
            }
            for seq, row in point_set.gdf.iterrows()
        ]
        session.execute(insert(CoastlinePointModel), rows_data)

        saved_id   = source.id
        saved_name = source.name

        session.commit()

    logger.success(
        f"Saved {len(rows_data)} points → "
        f"source id={saved_id} name={saved_name!r}"
    )


if __name__ == "__main__":
    main()
