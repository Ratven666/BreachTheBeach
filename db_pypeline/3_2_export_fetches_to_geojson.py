from __future__ import annotations

"""
db_pypeline/3_2_export_fetches_to_geojson.py

Экспортирует длины разгона ветра из БД в два GeoJSON-слоя:

    data/coastline/fetch/wind_fetch_points.geojson
        Точки береговой линии с атрибутами:
            point_id, lon, lat, seq,
            normal_azimuth_deg,
            fetch_count, fetch_min_m, fetch_max_m,
            fetch_mean_m, fetch_normal_m

    data/coastline/fetch/wind_fetch_rays.geojson
        Лучи (LineString) от точки в направлении азимута:
            point_id, seq, azimuth_deg, fetch_length_m,
            is_normal_dir

Конец луча вычисляется через pyproj.Geod.fwd() на эллипсоиде WGS-84,
что исключает геометрические артефакты сферической аппроксимации.

Запуск (из корня проекта):
    python db_pypeline/3_2_export_fetches_to_geojson.py
"""

import os
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------
# Конфигурация
# ------------------------------------------------------------------

DATABASE_PATH = Path("data/db/coastline.db")

# None → последний набор нормалей
NORMAL_SOURCE_ID: int | None = None

OUTPUT_DIR = Path("data/coastline/fetch")

# Экспортировать лучи (может быть крупным файлом при шаге 1°)
EXPORT_RAYS: bool = True

# Если True — экспортирует только луч нормали для каждой точки
RAYS_NORMAL_ONLY: bool = False


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------

def setup_env() -> None:
    if "COASTLINE_DATABASE_URL" not in os.environ:
        db_url = f"sqlite:///{DATABASE_PATH.resolve().as_posix()}"
        os.environ["COASTLINE_DATABASE_URL"] = db_url
        logger.debug(f"COASTLINE_DATABASE_URL → {db_url}")


def _azimuth_to_endpoint(
    lon: float, lat: float, azimuth_deg: float, length_m: float
) -> tuple[float, float]:
    """
    Вычисляет конечную точку луча на эллипсоиде WGS-84 (прямая геодезическая задача).
    Возвращает (end_lon, end_lat) в градусах.

    Использует pyproj.Geod — точнее сферической аппроксимации на 0.3–1%.
    Azimuth: bearing от севера по часовой стрелке (0 = N, 90 = E, ...).
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    end_lon, end_lat, _ = geod.fwd(lon, lat, azimuth_deg, length_m)
    return float(end_lon), float(end_lat)


def _angular_distance(a: float, b: float) -> float:
    """Минимальный угол между двумя азимутами [0, 180]."""
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def main() -> None:
    setup_env()

    import geopandas as gpd
    from collections import defaultdict
    from shapely.geometry import LineString, Point
    from sqlalchemy import select

    from src.coastline.storage import db as database
    from src.coastline.storage.models import (
        Base,
        CoastlineNormalModel,
        CoastlineNormalSourceModel,
        CoastlinePointModel,
        WindFetchModel,
    )

    Base.metadata.create_all(database.engine)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Определяем normal_source_id
    # ------------------------------------------------------------------
    with database.SessionLocal() as session:
        if NORMAL_SOURCE_ID is None:
            ns = (
                session.execute(
                    select(CoastlineNormalSourceModel).order_by(
                        CoastlineNormalSourceModel.id.desc()
                    )
                )
                .scalars()
                .first()
            )
        else:
            ns = session.get(CoastlineNormalSourceModel, NORMAL_SOURCE_ID)

        if ns is None:
            raise ValueError(
                "Набор нормалей не найден. "
                "Сначала запустите db_pypeline/2_1_normals_to_db.py"
            )

        normal_source_id = ns.id
        logger.info(
            f"Normal source: id={normal_source_id}, "
            f"name={ns.name!r}, sea_side={ns.sea_side}"
        )

        # ------------------------------------------------------------------
        # 2. Нормали: point_id → normal_azimuth_deg
        # ------------------------------------------------------------------
        normal_rows = (
            session.execute(
                select(CoastlineNormalModel).where(
                    CoastlineNormalModel.normal_source_id == normal_source_id
                )
            )
            .scalars()
            .all()
        )
        normal_by_point: dict[int, float] = {
            r.point_id: r.normal_azimuth_deg for r in normal_rows
        }
        logger.info(f"Нормалей загружено: {len(normal_by_point)}")

        # ------------------------------------------------------------------
        # 3. Точки береговой линии
        # ------------------------------------------------------------------
        point_ids = list(normal_by_point.keys())
        point_rows = (
            session.execute(
                select(CoastlinePointModel).where(
                    CoastlinePointModel.id.in_(point_ids)
                )
            )
            .scalars()
            .all()
        )
        point_by_id: dict[int, CoastlinePointModel] = {r.id: r for r in point_rows}
        logger.info(f"Точек загружено: {len(point_by_id)}")

        # ------------------------------------------------------------------
        # 4. Фетчи
        # ------------------------------------------------------------------
        fetch_rows = (
            session.execute(
                select(WindFetchModel)
                .where(WindFetchModel.point_id.in_(point_ids))
                .order_by(WindFetchModel.point_id, WindFetchModel.azimuth_deg)
            )
            .scalars()
            .all()
        )

    logger.info(f"Фетчей загружено: {len(fetch_rows)}")

    if not fetch_rows:
        logger.warning(
            "Таблица wind_fetches пуста. "
            "Сначала запустите db_pypeline/3_1_wind_fetch_to_db.py"
        )
        return

    # ------------------------------------------------------------------
    # 5. Группируем фетчи по точке
    # ------------------------------------------------------------------
    fetches_by_point: dict[int, list[WindFetchModel]] = defaultdict(list)
    for f in fetch_rows:
        fetches_by_point[f.point_id].append(f)

    # ------------------------------------------------------------------
    # 6. Слой точек с агрегированными атрибутами
    # ------------------------------------------------------------------
    point_records: list[dict] = []

    for pid, p in point_by_id.items():
        normal_az = normal_by_point.get(pid)
        fetches = fetches_by_point.get(pid, [])

        if fetches:
            lengths = [f.fetch_length_m for f in fetches]
            fetch_min = min(lengths)
            fetch_max = max(lengths)
            fetch_mean = sum(lengths) / len(lengths)
            fetch_count = len(lengths)

            if normal_az is not None:
                closest = min(
                    fetches,
                    key=lambda f: _angular_distance(f.azimuth_deg, normal_az),
                )
                fetch_normal_m = closest.fetch_length_m
            else:
                fetch_normal_m = None
        else:
            fetch_min = fetch_max = fetch_mean = fetch_normal_m = None
            fetch_count = 0

        point_records.append(
            {
                "point_id":           pid,
                "seq":                p.seq,
                "normal_azimuth_deg": normal_az,
                "fetch_count":        fetch_count,
                "fetch_min_m":        fetch_min,
                "fetch_max_m":        fetch_max,
                "fetch_mean_m":       round(fetch_mean, 2) if fetch_mean else None,
                "fetch_normal_m":     fetch_normal_m,
                "geometry":           Point(p.lon, p.lat),
            }
        )

    points_gdf = gpd.GeoDataFrame(
        point_records, geometry="geometry", crs="EPSG:4326"
    )
    points_path = OUTPUT_DIR / "wind_fetch_points.geojson"
    points_gdf.to_file(str(points_path), driver="GeoJSON")
    logger.success(f"Экспортировано {len(points_gdf)} точек → {points_path}")

    # ------------------------------------------------------------------
    # 7. Слой лучей
    # ------------------------------------------------------------------
    if not EXPORT_RAYS:
        logger.info("Экспорт лучей пропущен (EXPORT_RAYS=False)")
        return

    ray_records: list[dict] = []

    for pid, fetches in fetches_by_point.items():
        p = point_by_id.get(pid)
        if p is None:
            continue

        normal_az = normal_by_point.get(pid)

        # Ближайший к нормали азимут — вычисляем один раз на точку
        if normal_az is not None:
            normal_closest_az = min(
                fetches,
                key=lambda f: _angular_distance(f.azimuth_deg, normal_az),
            ).azimuth_deg
        else:
            normal_closest_az = None

        for f in fetches:
            if RAYS_NORMAL_ONLY:
                if normal_closest_az is None:
                    continue
                if _angular_distance(f.azimuth_deg, normal_closest_az) > 0.5:
                    continue

            # Прямая геодезическая задача на WGS-84 — нет артефактов
            # сферической аппроксимации для точек на краях береговой линии
            end_lon, end_lat = _azimuth_to_endpoint(
                lon=p.lon,
                lat=p.lat,
                azimuth_deg=f.azimuth_deg,
                length_m=f.fetch_length_m,
            )

            is_normal = (
                normal_closest_az is not None
                and abs(f.azimuth_deg - normal_closest_az) < 1e-6
            )

            ray_records.append(
                {
                    "point_id":       pid,
                    "seq":            p.seq,
                    "azimuth_deg":    f.azimuth_deg,
                    "fetch_length_m": f.fetch_length_m,
                    "is_normal_dir":  is_normal,
                    "geometry":       LineString(
                        [(p.lon, p.lat), (end_lon, end_lat)]
                    ),
                }
            )

    rays_gdf = gpd.GeoDataFrame(
        ray_records, geometry="geometry", crs="EPSG:4326"
    )
    rays_path = OUTPUT_DIR / "wind_fetch_rays.geojson"
    rays_gdf.to_file(str(rays_path), driver="GeoJSON")
    logger.success(f"Экспортировано {len(rays_gdf)} лучей → {rays_path}")


if __name__ == "__main__":
    main()
