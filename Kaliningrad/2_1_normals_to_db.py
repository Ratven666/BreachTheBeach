from __future__ import annotations

from coastline.storage.models import Base, CoastlineSourceModel, CoastlinePointModel, CoastlineNormalSourceModel, \
    CoastlineNormalModel

"""
db_pypeline/2_1_normals_to_db.py

Запуск (из корня проекта):
    python db_pypeline/2_1_normals_to_db.py
"""

import os
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

# ── Конфигурация ──────────────────────────────────────────────────────
DATABASE_PATH = Path("data/db/coastline.db")

POINT_SOURCE_ID: int | None = None   # None → последний по id

NORMALS_NAME    = "kaliningrad_coastline_step200m_right"
SEA_SIDE        = "right"
TANGENT_DELTA_M = 0.1
# ─────────────────────────────────────────────────────────────────────


def setup_env() -> None:
    if "COASTLINE_DATABASE_URL" not in os.environ:
        url = f"sqlite:///{DATABASE_PATH.resolve().as_posix()}"
        os.environ["COASTLINE_DATABASE_URL"] = url
        logger.debug(f"COASTLINE_DATABASE_URL → {url}")


def main() -> None:
    setup_env()

    from sqlalchemy import insert, select

    import geopandas as gpd
    from shapely.geometry import Point

    from src.coastline.domain.CoastlineDataset import CoastlineDataset
    from src.coastline.domain.CoastlinePointSet import CoastlinePointSet, PointSetMeta
    from src.coastline.services.CoastlineNormalService import (
        CoastlineNormalConfig,
        CoastlineNormalService,
    )
    from src.coastline.storage import db as _db


    Base.metadata.create_all(_db.engine)
    logger.info("DB schema ready")

    # 1. Источник точек + пути из метаданных
    with _db.SessionLocal() as session:
        if POINT_SOURCE_ID is not None:
            source = session.get(CoastlineSourceModel, POINT_SOURCE_ID)
            if source is None:
                raise ValueError(f"CoastlineSource id={POINT_SOURCE_ID} not found")
        else:
            source = session.execute(
                select(CoastlineSourceModel).order_by(CoastlineSourceModel.id.desc())
            ).scalars().first()
            if source is None:
                raise ValueError(
                    "Нет записей в coastline_sources. "
                    "Сначала запустите 1_1_import_coastline_points.py"
                )

        point_source_id    = source.id
        main_geojson_path  = source.main_geojson_path
        other_geojson_path = source.other_geojson_path
        source_crs         = source.crs

        logger.info(
            f"Point source: id={point_source_id}  name={source.name!r}  "
            f"pts={source.points_count}"
        )
        logger.info(f"  main_geojson  → {main_geojson_path}")
        logger.info(f"  other_geojson → {other_geojson_path}")

        if not main_geojson_path:
            raise ValueError(
                "main_geojson_path пустой. "
                "Перезапустите 1_1_import_coastline_points.py."
            )

        # 2. Загружаем точки — нужны id для FK в нормалях
        pt_rows = session.execute(
            select(CoastlinePointModel)
            .where(CoastlinePointModel.source_id == point_source_id)
            .order_by(CoastlinePointModel.seq)
        ).scalars().all()

        # seq → (db_id, lon, lat)
        seq_to_point = {
            r.seq: (r.id, r.lon, r.lat)
            for r in pt_rows
        }

    logger.info(f"Loaded {len(seq_to_point)} points from DB")

    # 3. Восстанавливаем GeoDataFrame для сервиса нормалей
    records = [
        {"seq": seq, "geometry": Point(lon, lat)}
        for seq, (_, lon, lat) in sorted(seq_to_point.items())
    ]
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=source_crs)
    point_set = CoastlinePointSet(
        gdf=gdf,
        meta=PointSetMeta(
            name=NORMALS_NAME,
            source_dataset_name="",
            strategy_name="",
            source_mode="",
            points_count=len(gdf),
        ),
    )

    # 4. Датасет береговой линии из путей в метаданных БД
    dataset = CoastlineDataset.from_geojson(
        main_path=main_geojson_path,
        other_path=other_geojson_path,
        name=NORMALS_NAME,
    )
    logger.info(
        f"Dataset ready: main={len(dataset.main_gdf)} feat, "
        f"metric_crs={dataset.metric_crs}"
    )

    # 5. Расчёт нормалей
    config = CoastlineNormalConfig(
        sea_side=SEA_SIDE,
        tangent_delta_m=TANGENT_DELTA_M,
        working_crs=str(dataset.metric_crs),
    )
    normal_set = CoastlineNormalService(config).build_points_with_normals(
        point_set=point_set,
        dataset=dataset,
        name=NORMALS_NAME,
    )
    logger.info(f"Calculated {len(normal_set.gdf)} normals")

    report = normal_set.validate_vectors()
    if not report.is_valid:
        logger.warning(f"Validation issues: {report}")

    # 6. Сохраняем в БД
    with _db.SessionLocal() as session:

        ns = CoastlineNormalSourceModel(
            name=NORMALS_NAME,
            point_source_id=point_source_id,
            sea_side=SEA_SIDE,
            tangent_delta_m=TANGENT_DELTA_M,
            working_crs=str(config.working_crs) if config.working_crs else None,
            result_crs=str(normal_set.gdf.crs) if normal_set.gdf.crs else None,
            normals_count=len(normal_set.gdf),
        )
        session.add(ns)
        session.flush()

        rows_data = [
            {
                "normal_source_id":    ns.id,
                "point_id":           seq_to_point[int(seq)][0],
                "nx":                 float(row["nx"]),
                "ny":                 float(row["ny"]),
                "normal_azimuth_deg": float(row["normal_azimuth_deg"]),
            }
            for seq, row in normal_set.gdf.iterrows()
        ]
        session.execute(insert(CoastlineNormalModel), rows_data)

        saved_ns_id = ns.id
        session.commit()

    logger.success(
        f"Done. Saved {len(rows_data)} normals → "
        f"normal_source id={saved_ns_id} name={NORMALS_NAME!r}"
    )


if __name__ == "__main__":
    main()
