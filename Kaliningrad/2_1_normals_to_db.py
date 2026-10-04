from __future__ import annotations

"""
Вычисляет нормали относительно явно указанной береговой линии
и сохраняет их в SQLite.

Направление линий НЕ меняется: left/right определяются порядком вершин
в GeoJSON. Направление нужно подготовить заранее (например, в QGIS).

Примеры запуска из корня проекта:

    python db_pypeline/2_1_normals_to_db.py \
        --coastline data/coastline/kaliningrad_reference.geojson

    python db_pypeline/2_1_normals_to_db.py \
        --coastline /absolute/path/coastline.geojson \
        --point-source-id 3 --sea-side right \
        --tangent-delta-m 30 --max-reference-distance-m 500

Если --coastline не задан, используется main_geojson_path источника точек.
Относительные пути разрешаются относительно корня проекта.
"""

import argparse
import math
import os
import sys
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Конфигурация ──────────────────────────────────────────────────────
DATABASE_PATH = PROJECT_ROOT / "Kaliningrad" / "data" / "db" / "coastline.db"

POINT_SOURCE_ID: int | None = None
COASTLINE_MAIN_PATH: Path | None = PROJECT_ROOT / "Kaliningrad" / "data" / "coastline" / "kaliningrad_OFFSET_merged.geojson"

NORMALS_NAME = "kaliningrad_coastline_step200m_right"
SEA_SIDE = "right"
TANGENT_DELTA_M = 20.0
MAX_REFERENCE_DISTANCE_M: float | None = 50.0
# ─────────────────────────────────────────────────────────────────────


def setup_env() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if "COASTLINE_DATABASE_URL" not in os.environ:
        url = f"sqlite:///{DATABASE_PATH.resolve().as_posix()}"
        os.environ["COASTLINE_DATABASE_URL"] = url
        logger.debug(f"COASTLINE_DATABASE_URL -> {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate coastline normals using an explicitly selected "
            "reference GeoJSON. Line direction is never modified."
        )
    )
    parser.add_argument(
        "--coastline",
        type=Path,
        default=COASTLINE_MAIN_PATH,
        help=(
            "Reference coastline GeoJSON. Relative paths are resolved "
            "from the project root. If omitted, the path stored in the "
            "point-source metadata is used."
        ),
    )
    parser.add_argument(
        "--point-source-id",
        type=int,
        default=POINT_SOURCE_ID,
        help="coastline_sources.id; default: latest source",
    )
    parser.add_argument("--name", default=NORMALS_NAME)
    parser.add_argument(
        "--sea-side", choices=("left", "right"), default=SEA_SIDE
    )
    parser.add_argument(
        "--tangent-delta-m", type=float, default=TANGENT_DELTA_M
    )
    parser.add_argument(
        "--max-reference-distance-m",
        type=float,
        default=MAX_REFERENCE_DISTANCE_M,
        help="Maximum point-to-reference distance in metres.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path, *, description: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{description} is not a file: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    setup_env()

    import geopandas as gpd
    from pyproj import CRS
    from shapely.geometry import Point
    from sqlalchemy import insert, select

    from src.coastline.domain.CoastlineDataset import CoastlineDataset
    from src.coastline.domain.CoastlinePointSet import (
        CoastlinePointSet,
        PointSetMeta,
    )
    from src.coastline.services.CoastlineNormalService import (
        CoastlineNormalConfig,
        CoastlineNormalService,
    )
    from src.coastline.storage import db as _db
    from src.coastline.storage.models import (
        Base,
        CoastlineNormalModel,
        CoastlineNormalSourceModel,
        CoastlinePointModel,
        CoastlineSourceModel,
    )

    Base.metadata.create_all(_db.engine)
    logger.info(f"Database: {DATABASE_PATH}")
    logger.info("DB schema ready")

    with _db.SessionLocal() as session:
        if args.point_source_id is not None:
            source = session.get(CoastlineSourceModel, args.point_source_id)
            if source is None:
                raise ValueError(
                    f"CoastlineSource id={args.point_source_id} not found"
                )
        else:
            source = session.execute(
                select(CoastlineSourceModel).order_by(
                    CoastlineSourceModel.id.desc()
                )
            ).scalars().first()
            if source is None:
                raise ValueError(
                    "No records in coastline_sources. "
                    "Run 1_1_import_coastline_points.py first."
                )

        point_source_id = int(source.id)
        source_name = str(source.name)
        source_crs = source.crs
        metadata_main_path = source.main_geojson_path
        expected_points_count = int(source.points_count)

        point_rows = session.execute(
            select(CoastlinePointModel)
            .where(CoastlinePointModel.source_id == point_source_id)
            .order_by(CoastlinePointModel.seq)
        ).scalars().all()

    if not point_rows:
        raise ValueError(
            f"Point source id={point_source_id} contains no points"
        )

    if len(point_rows) != expected_points_count:
        raise ValueError(
            f"Point count mismatch for source id={point_source_id}: "
            f"metadata={expected_points_count}, rows={len(point_rows)}"
        )

    seq_values = [int(r.seq) for r in point_rows]
    if len(seq_values) != len(set(seq_values)):
        raise ValueError(
            f"Point source id={point_source_id} contains duplicate seq values"
        )

    point_ids = [int(r.id) for r in point_rows]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("Duplicate coastline_points.id values detected")

    if source_crs is None:
        raise ValueError(f"Point source id={point_source_id} has no CRS")

    if CRS.from_user_input(source_crs).to_epsg() != 4326:
        raise ValueError(
            f"Point source CRS is {source_crs!r}, but database lon/lat "
            "must be EPSG:4326. Re-import the points with the corrected "
            "1_1_import_coastline_points.py."
        )

    selected_path = args.coastline or metadata_main_path
    if not selected_path:
        raise ValueError(
            "Reference coastline path is not specified. "
            "Pass --coastline PATH or re-import point-source metadata."
        )

    reference_path = resolve_path(
        selected_path, description="Reference coastline GeoJSON"
    )

    logger.info(
        f"Point source: id={point_source_id}, name={source_name!r}, "
        f"points={len(point_rows)}"
    )
    logger.info(f"Reference coastline: {reference_path}")

    points_gdf = gpd.GeoDataFrame(
        [
            {
                "seq": int(r.seq),
                "point_id": int(r.id),
                "geometry": Point(float(r.lon), float(r.lat)),
            }
            for r in point_rows
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    point_set = CoastlinePointSet(
        gdf=points_gdf,
        meta=PointSetMeta(
            name=args.name,
            source_dataset_name=source_name,
            strategy_name="database",
            source_mode="main",
            points_count=len(points_gdf),
        ),
    )

    dataset = CoastlineDataset.from_geojson(
        main_path=reference_path,
        other_path=None,
        name=args.name,
    )

    config = CoastlineNormalConfig(
        sea_side=args.sea_side,
        tangent_delta_m=args.tangent_delta_m,
        working_crs=str(dataset.metric_crs),
        max_reference_distance_m=args.max_reference_distance_m,
    )

    normal_set = CoastlineNormalService(config).build_points_with_normals(
        point_set=point_set,
        dataset=dataset,
        name=args.name,
    )

    if len(normal_set.gdf) != len(points_gdf):
        raise RuntimeError(
            f"Normal count mismatch: points={len(points_gdf)}, "
            f"normals={len(normal_set.gdf)}"
        )

    report = normal_set.validate_vectors()
    logger.info(
        "Validation: "
        f"valid={report.is_valid}, "
        f"invalid_tangent={report.invalid_tangent_count}, "
        f"invalid_normal={report.invalid_normal_count}, "
        f"invalid_orthogonality={report.invalid_orthogonality_count}, "
        f"invalid_side={report.invalid_side_count}, "
        f"invalid_azimuth={report.invalid_azimuth_count}, "
        f"invalid_azimuth_consistency="
        f"{report.invalid_azimuth_consistency_count}, "
        f"max_abs_dot={report.max_abs_dot_product}, "
        f"chainage_not_sorted(info)={report.chainage_not_sorted}"
    )

    if not report.is_valid:
        raise RuntimeError(
            "Normal vector validation failed. "
            "Nothing has been written to the database.\n"
            + normal_set.debug_report(n=10)
        )

    normal_point_ids = [int(v) for v in normal_set.gdf["point_id"]]

    if len(normal_point_ids) != len(set(normal_point_ids)):
        raise RuntimeError(
            "Calculated normal set contains duplicate point_id values"
        )

    if set(normal_point_ids) != set(point_ids):
        missing = sorted(set(point_ids) - set(normal_point_ids))
        foreign = sorted(set(normal_point_ids) - set(point_ids))
        raise RuntimeError(
            "Calculated normals do not match the selected point source: "
            f"missing_point_ids={missing[:20]}, "
            f"foreign_point_ids={foreign[:20]}"
        )

    rows_data: list[dict] = []
    for _, row in normal_set.gdf.iterrows():
        values = {
            "normal_source_id": None,
            "point_id": int(row["point_id"]),
            "nx": float(row["nx"]),
            "ny": float(row["ny"]),
            "normal_azimuth_deg": float(row["normal_azimuth_deg"]),
        }
        if not all(
            math.isfinite(values[c])
            for c in ("nx", "ny", "normal_azimuth_deg")
        ):
            raise RuntimeError(
                f"Non-finite normal values for point_id={values['point_id']}"
            )
        rows_data.append(values)

    with _db.SessionLocal() as session:
        normal_source = CoastlineNormalSourceModel(
            name=args.name,
            point_source_id=point_source_id,
            sea_side=args.sea_side,
            tangent_delta_m=float(args.tangent_delta_m),
            working_crs=str(normal_set.gdf.crs),
            result_crs=str(normal_set.gdf.crs),
            normals_count=len(rows_data),
        )
        session.add(normal_source)
        session.flush()

        for row in rows_data:
            row["normal_source_id"] = int(normal_source.id)

        session.execute(insert(CoastlineNormalModel), rows_data)

        saved_source_id = int(normal_source.id)
        session.commit()

    logger.success(
        f"Saved {len(rows_data)} normals: "
        f"normal_source_id={saved_source_id}, name={args.name!r}, "
        f"reference={reference_path}"
    )


if __name__ == "__main__":
    main()
