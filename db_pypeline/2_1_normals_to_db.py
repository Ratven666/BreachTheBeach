from __future__ import annotations

"""
db_pypeline/2_1_normals_to_db.py

Запуск (из корня проекта):
    python db_pypeline/2_1_normals_to_db.py
    python db_pypeline/2_1_normals_to_db.py --coastline data/coastline/ref.geojson
    python db_pypeline/2_1_normals_to_db.py --coastline ref.geojson \
        --point-source-id 3 --sea-side right --tangent-delta-m 5 \
        --max-reference-distance-m 500

Без --coastline используется main_geojson_path из метаданных источника точек.
Относительные пути разрешаются от корня проекта.
"""

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.coastline.storage.models import (  # noqa: E402
    Base,
    CoastlineNormalModel,
    CoastlineNormalSourceModel,
    CoastlinePointModel,
    CoastlineSourceModel,
)

# ── Конфигурация по умолчанию ─────────────────────────────────────────
DATABASE_PATH = PROJECT_ROOT / "db_pypeline" / "data" / "db" / "coastline.db"

POINT_SOURCE_ID: int | None = None          # None → последний по id
COASTLINE_MAIN_PATH: Path | None = None     # None → путь из метаданных БД

NORMALS_NAME = "nvrsk_normals_step200m_right"
SEA_SIDE = "right"
TANGENT_DELTA_M = 5.0
MAX_REFERENCE_DISTANCE_M: float | None = 500.0
# ─────────────────────────────────────────────────────────────────────


def setup_env() -> None:
    if "COASTLINE_DATABASE_URL" not in os.environ:
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{DATABASE_PATH.resolve().as_posix()}"
        os.environ["COASTLINE_DATABASE_URL"] = url
        logger.debug(f"COASTLINE_DATABASE_URL → {url}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normals to coastline → SQLite")
    p.add_argument("--coastline", type=Path, default=COASTLINE_MAIN_PATH,
                   help="GeoJSON береговой линии, относительно которой "
                        "считаются нормали")
    p.add_argument("--point-source-id", type=int, default=POINT_SOURCE_ID)
    p.add_argument("--name", default=NORMALS_NAME)
    p.add_argument("--sea-side", choices=("left", "right"), default=SEA_SIDE)
    p.add_argument("--tangent-delta-m", type=float, default=TANGENT_DELTA_M)
    p.add_argument("--max-reference-distance-m", type=float,
                   default=MAX_REFERENCE_DISTANCE_M)
    return p.parse_args()


def resolve_path(path: str | Path, description: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{description} not found: {p}")
    return p


def main() -> None:
    args = parse_args()
    setup_env()

    import geopandas as gpd
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

    Base.metadata.create_all(_db.engine)
    logger.info(f"DB: {os.environ['COASTLINE_DATABASE_URL']}")

    # 1. Источник точек
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
                    "Нет записей в coastline_sources. "
                    "Сначала запустите 1_1_import_coastline_points.py"
                )

        point_source_id = source.id
        metadata_main_path = source.main_geojson_path
        other_geojson_path = source.other_geojson_path
        source_crs = source.crs

        logger.info(
            f"Point source: id={point_source_id}  name={source.name!r}  "
            f"pts={source.points_count}  crs={source_crs}"
        )

        pt_rows = session.execute(
            select(CoastlinePointModel)
            .where(CoastlinePointModel.source_id == point_source_id)
            .order_by(CoastlinePointModel.seq)
        ).scalars().all()

        points_data = [(r.id, r.seq, r.lon, r.lat) for r in pt_rows]

    if not points_data:
        raise ValueError(f"Источник {point_source_id} не содержит точек")
    if source_crs is None:
        raise ValueError(f"У источника {point_source_id} не задана CRS")

    seqs = [seq for _, seq, _, _ in points_data]
    if len(seqs) != len(set(seqs)):
        raise ValueError("В источнике точек есть повторяющиеся seq")

    logger.info(f"Loaded {len(points_data)} points from DB")

    # 2. Береговая линия: явный путь либо метаданные БД
    selected = args.coastline or metadata_main_path
    if not selected:
        raise ValueError(
            "Не указан путь к береговой линии: задайте --coastline "
            "или перезапустите 1_1_import_coastline_points.py"
        )
    main_path = resolve_path(selected, "Reference coastline GeoJSON")
    other_path = (
        resolve_path(other_geojson_path, "Other coastline GeoJSON")
        if other_geojson_path else None
    )
    logger.info(f"Reference coastline → {main_path}")

    # 3. GeoDataFrame точек; point_id (PK из БД) и seq передаются в сервис
    records = [
        {"seq": int(seq), "point_id": int(pid), "geometry": Point(lon, lat)}
        for pid, seq, lon, lat in points_data
    ]
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=source_crs)
    point_set = CoastlinePointSet(
        gdf=gdf,
        meta=PointSetMeta(
            name=args.name,
            source_dataset_name="",
            strategy_name="",
            source_mode="",
            points_count=len(gdf),
        ),
    )

    # 4. Датасет
    dataset = CoastlineDataset.from_geojson(
        main_path=str(main_path),
        other_path=str(other_path) if other_path else None,
        name=args.name,
    )
    logger.info(
        f"Dataset ready: main={len(dataset.main_gdf)} feat, "
        f"metric_crs={dataset.metric_crs}"
    )

    # 5. Расчёт нормалей
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
    logger.info(f"Calculated {len(normal_set.gdf)} normals")

    report = normal_set.validate_vectors()
    if not report.is_valid:
        raise RuntimeError(
            f"Validation failed, ничего не записано в БД: {report}"
        )

    # 6. Проверка соответствия нормалей точкам БД
    ngdf = normal_set.gdf
    if len(ngdf) != len(points_data):
        raise RuntimeError(
            f"Число нормалей ({len(ngdf)}) != числу точек ({len(points_data)})"
        )

    db_point_ids = {pid for pid, _, _, _ in points_data}
    seq_to_pid = {seq: pid for pid, seq, _, _ in points_data}

    rows_data = []
    for _, row in ngdf.iterrows():
        pid = int(row["point_id"])
        if pid not in db_point_ids:
            raise RuntimeError(f"point_id={pid} отсутствует в БД")
        if "seq" in ngdf.columns and seq_to_pid[int(row["seq"])] != pid:
            raise RuntimeError(
                f"Несогласованность seq={int(row['seq'])} и point_id={pid}"
            )
        rows_data.append(
            {
                "point_id": pid,
                "nx": float(row["nx"]),
                "ny": float(row["ny"]),
                "normal_azimuth_deg": float(row["normal_azimuth_deg"]),
            }
        )

    if len({r["point_id"] for r in rows_data}) != len(rows_data):
        raise RuntimeError("Дублирующиеся point_id среди нормалей")

    # 7. Запись одной транзакцией
    with _db.SessionLocal() as session:
        ns = CoastlineNormalSourceModel(
            name=args.name,
            point_source_id=point_source_id,
            sea_side=args.sea_side,
            tangent_delta_m=args.tangent_delta_m,
            working_crs=str(config.working_crs) if config.working_crs else None,
            result_crs=str(ngdf.crs) if ngdf.crs else None,
            normals_count=len(rows_data),
        )
        session.add(ns)
        session.flush()

        for r in rows_data:
            r["normal_source_id"] = ns.id
        session.execute(insert(CoastlineNormalModel), rows_data)

        saved_ns_id = ns.id
        session.commit()

    logger.success(
        f"Done. Saved {len(rows_data)} normals → "
        f"normal_source id={saved_ns_id} name={args.name!r}"
    )


if __name__ == "__main__":
    main()