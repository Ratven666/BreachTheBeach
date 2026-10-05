from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.coastline.storage.models import (  # noqa: E402
    Base,
    CoastlinePointModel,
    CoastlineSourceModel,
)

# ── Конфигурация ──────────────────────────────────────────────────────
DATABASE_PATH = PROJECT_ROOT / "db_pypeline" / "data" / "db" / "coastline.db"

COASTLINE_MAIN_PATH = Path("data/coastline/nvrsk_main_coastline.geojson")
COASTLINE_OTHER_PATH: Path | None = None

DATASET_NAME = "nvrsk_main_coastline"
STRATEGY_STEP_M = 200.0
INPUT_CRS = "EPSG:4326"
# ─────────────────────────────────────────────────────────────────────


def _abs(path: Path) -> Path:
    p = path.expanduser()
    return (p if p.is_absolute() else PROJECT_ROOT / p).resolve()


def setup_env() -> None:
    if "COASTLINE_DATABASE_URL" not in os.environ:
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
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

    main_path = _abs(COASTLINE_MAIN_PATH)
    other_path = _abs(COASTLINE_OTHER_PATH) if COASTLINE_OTHER_PATH else None
    if not main_path.is_file():
        raise FileNotFoundError(f"Main coastline not found: {main_path}")

    Base.metadata.create_all(_db.engine)
    logger.info(f"DB: {os.environ['COASTLINE_DATABASE_URL']}")

    dataset = CoastlineDataset.from_geojson(
        main_path=str(main_path),
        other_path=str(other_path) if other_path else None,
        name=DATASET_NAME,
    )
    logger.info(
        f"Dataset: main={len(dataset.main_gdf)} feat, "
        f"other={len(dataset.other_gdf)} feat, crs={dataset.crs}"
    )

    strategy = EqualStepAlongLineStrategy(
        step_m=STRATEGY_STEP_M,
        source=PointSource.MAIN_ONLY,
        include_endpoints=True,
        working_crs=None,
        input_crs=INPUT_CRS,
    )
    strategy_params = json.dumps(
        {
            "step_m": STRATEGY_STEP_M,
            "source": PointSource.MAIN_ONLY.value,
            "include_endpoints": True,
            "working_crs": None,
            "input_crs": INPUT_CRS,
        },
        ensure_ascii=False,
    )

    point_set = CoastlinePointExtractor().extract(
        dataset=dataset, strategy=strategy, name=DATASET_NAME
    )
    point_set.print_summary()

    with _db.SessionLocal() as session:
        source = CoastlineSourceModel(
            name=point_set.meta.name,
            strategy_name=point_set.meta.strategy_name,
            source_mode=point_set.meta.source_mode,
            strategy_params=strategy_params,
            main_geojson_path=str(main_path),
            other_geojson_path=str(other_path) if other_path else None,
            crs=str(point_set.gdf.crs) if point_set.gdf.crs else None,
            points_count=len(point_set.gdf),
        )
        session.add(source)
        session.flush()

        rows_data = [
            {
                "source_id": source.id,
                "seq": int(seq),
                "lon": float(row.geometry.x),
                "lat": float(row.geometry.y),
            }
            for seq, row in point_set.gdf.iterrows()
        ]
        session.execute(insert(CoastlinePointModel), rows_data)

        saved_id, saved_name = source.id, source.name
        session.commit()

    logger.success(
        f"Saved {len(rows_data)} points → "
        f"source id={saved_id} name={saved_name!r}"
    )


if __name__ == "__main__":
    main()