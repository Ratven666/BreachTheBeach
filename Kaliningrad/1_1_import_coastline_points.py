from __future__ import annotations

"""
Импортирует точки береговой линии в SQLite.

Контракт хранения:
- coastline_points.lon и coastline_points.lat всегда EPSG:4326;
- coastline_sources.crs всегда содержит EPSG:4326;
- относительные пути разрешаются относительно корня проекта;
- используется та же база data/db/coastline.db, что и на этапе нормалей.
"""

import json
import os
import sys
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Конфигурация ──────────────────────────────────────────────────────
DATABASE_PATH = PROJECT_ROOT / "Kaliningrad" / "data" / "db" / "coastline.db"

COASTLINE_MAIN_PATH = Path(
    "Kaliningrad/data/coastline/KaliningradOSM_merged.geojson"
)
COASTLINE_OTHER_PATH: Path | None = None

DATASET_NAME = "KaliningradOSM"
STRATEGY_STEP_M = 200.0
INPUT_CRS = "EPSG:4326"
DATABASE_POINT_CRS = "EPSG:4326"
# ─────────────────────────────────────────────────────────────────────


def setup_env() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if "COASTLINE_DATABASE_URL" not in os.environ:
        url = f"sqlite:///{DATABASE_PATH.resolve().as_posix()}"
        os.environ["COASTLINE_DATABASE_URL"] = url
        logger.debug(f"COASTLINE_DATABASE_URL -> {url}")


def resolve_path(
    path: str | Path,
    *,
    description: str,
) -> Path:
    resolved = Path(path).expanduser()

    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved

    resolved = resolved.resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {resolved}"
        )

    if not resolved.is_file():
        raise ValueError(
            f"{description} is not a file: {resolved}"
        )

    return resolved


def main() -> None:
    setup_env()

    from shapely.geometry import Point
    from sqlalchemy import insert

    from src.coastline.domain.CoastlineDataset import CoastlineDataset
    from src.coastline.point_strategies import EqualStepAlongLineStrategy
    from src.coastline.point_strategies.PointExtractionStrategy import (
        PointSource,
    )
    from src.coastline.services.CoastlinePointExtractor import (
        CoastlinePointExtractor,
    )
    from src.coastline.storage import db as _db
    from src.coastline.storage.models import (
        Base,
        CoastlinePointModel,
        CoastlineSourceModel,
    )

    main_path = resolve_path(
        COASTLINE_MAIN_PATH,
        description="Main coastline GeoJSON",
    )

    other_path = (
        resolve_path(
            COASTLINE_OTHER_PATH,
            description="Other coastline GeoJSON",
        )
        if COASTLINE_OTHER_PATH is not None
        else None
    )

    Base.metadata.create_all(_db.engine)
    logger.info(f"Database: {DATABASE_PATH}")
    logger.info("DB schema ready")

    dataset = CoastlineDataset.from_geojson(
        main_path=main_path,
        other_path=other_path,
        name=DATASET_NAME,
    )

    strategy = EqualStepAlongLineStrategy(
        step_m=STRATEGY_STEP_M,
        source=PointSource.MAIN_ONLY,
        include_endpoints=True,
        working_crs=None,
        input_crs=INPUT_CRS,
    )

    point_set = CoastlinePointExtractor().extract(
        dataset=dataset,
        strategy=strategy,
        name=DATASET_NAME,
    )

    if point_set.gdf.empty:
        raise ValueError("Point extraction produced no points")

    if point_set.gdf.crs is None:
        raise ValueError("Extracted point set has no CRS")

    # Независимо от CRS исходного GeoJSON координаты в SQLite
    # всегда записываются как lon/lat в WGS84.
    points_wgs84 = point_set.gdf.to_crs(
        DATABASE_POINT_CRS
    ).reset_index(drop=True)

    if not all(
        isinstance(geometry, Point)
        for geometry in points_wgs84.geometry
    ):
        raise TypeError("Extracted geometries must be Point")

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

    rows_data = [
        {
            "seq": int(seq),
            "lon": float(row.geometry.x),
            "lat": float(row.geometry.y),
        }
        for seq, (_, row) in enumerate(points_wgs84.iterrows())
    ]

    with _db.SessionLocal() as session:
        source = CoastlineSourceModel(
            name=point_set.meta.name,
            strategy_name=point_set.meta.strategy_name,
            source_mode=point_set.meta.source_mode,
            strategy_params=strategy_params,
            main_geojson_path=str(main_path),
            other_geojson_path=(
                str(other_path)
                if other_path is not None
                else None
            ),
            crs=DATABASE_POINT_CRS,
            points_count=len(rows_data),
        )

        session.add(source)
        session.flush()

        for row in rows_data:
            row["source_id"] = int(source.id)

        session.execute(
            insert(CoastlinePointModel),
            rows_data,
        )

        saved_id = int(source.id)
        saved_name = str(source.name)
        session.commit()

    logger.success(
        f"Saved {len(rows_data)} points: "
        f"source_id={saved_id}, name={saved_name!r}"
    )


if __name__ == "__main__":
    main()
