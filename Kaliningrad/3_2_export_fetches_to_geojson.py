from __future__ import annotations

"""
db_pypeline/3_2_export_fetches_to_geojson.py

Экспортирует длины разгона ветра из БД в два GeoJSON-слоя:

    <output-dir>/wind_fetch_points.geojson
        Точки береговой линии с атрибутами:
            point_id, seq, normal_azimuth_deg,
            fetch_count, fetch_min_m, fetch_max_m,
            fetch_mean_m, fetch_normal_m

        Статистика считается по ВСЕМ фетчам из БД
        и не зависит от шага экспортируемых лучей.

    <output-dir>/wind_fetch_rays.geojson
        Лучи (LineString) от точки в направлении азимута:
            point_id, seq, azimuth_deg, fetch_length_m,
            is_normal_dir

Шаг лучей:
    RAY_STEP_DEG = None  → экспортируются все лучи, имеющиеся в БД.
    RAY_STEP_DEG = 10.0  → экспортируются лучи с азимутами
                           RAY_ORIGIN_DEG + k * 10°.

Шаг применяется только к экспорту и не пересчитывает фетчи.
Он должен быть кратен шагу, с которым фетчи записаны в БД
(например, при БД-шаге 1° подходят 2°, 5°, 10°, 15°, 30°).
Луч нормали по умолчанию добавляется всегда (RAYS_ALWAYS_INCLUDE_NORMAL).

Конец луча вычисляется через pyproj.Geod.fwd() на эллипсоиде WGS-84.

Примеры запуска (из корня проекта):
    python db_pypeline/3_2_export_fetches_to_geojson.py
    python db_pypeline/3_2_export_fetches_to_geojson.py --ray-step-deg 10
    python db_pypeline/3_2_export_fetches_to_geojson.py \
        --ray-step-deg 15 --ray-origin-deg 0 --normal-source-id 3
    python db_pypeline/3_2_export_fetches_to_geojson.py --normal-only
    python db_pypeline/3_2_export_fetches_to_geojson.py --no-rays
"""

import argparse
import math
import os
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------
# Конфигурация
# ------------------------------------------------------------------

DATABASE_PATH = Path("/Users/mihailvystrcil/Documents/My Programms/BreachTheBeach/Kaliningrad/data/db/coastline.db")

# None → последний набор нормалей
NORMAL_SOURCE_ID: int | None = None

OUTPUT_DIR = Path("/Users/mihailvystrcil/Documents/My Programms/BreachTheBeach/Kaliningrad/data/coastline/fetch")

# Экспортировать лучи (при шаге 1° файл может быть крупным)
EXPORT_RAYS: bool = True

# True → экспортируется только луч нормали для каждой точки
RAYS_NORMAL_ONLY: bool = False

# Явный шаг лучей при экспорте, градусы. None → все лучи из БД.
# RAY_STEP_DEG: float | None = None
RAY_STEP_DEG: float | None = 20

# Азимут, от которого отсчитывается сетка шага (0 = север).
RAY_ORIGIN_DEG: float = 0.0

# Всегда добавлять в слой лучей луч, ближайший к нормали,
# даже если он не попадает в сетку RAY_STEP_DEG.
RAYS_ALWAYS_INCLUDE_NORMAL: bool = True

# Допуск при проверке попадания азимута в сетку шага, градусы.
RAY_STEP_TOLERANCE_DEG: float = 1e-6

# Допуск при определении «луча нормали», градусы.
NORMAL_RAY_TOLERANCE_DEG: float = 1e-6


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Экспорт длин разгона ветра из БД в GeoJSON."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="Путь к SQLite-базе (если не задана COASTLINE_DATABASE_URL).",
    )
    parser.add_argument(
        "--normal-source-id",
        type=int,
        default=NORMAL_SOURCE_ID,
        help="coastline_normal_sources.id; по умолчанию последний.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Каталог результата.",
    )
    parser.add_argument(
        "--ray-step-deg",
        type=float,
        default=RAY_STEP_DEG,
        help="Шаг лучей при экспорте, градусы. Без параметра — все лучи.",
    )
    parser.add_argument(
        "--ray-origin-deg",
        type=float,
        default=RAY_ORIGIN_DEG,
        help="Азимут начала сетки шага лучей, градусы.",
    )
    parser.add_argument(
        "--no-rays",
        action="store_false",
        dest="export_rays",
        default=EXPORT_RAYS,
        help="Не экспортировать слой лучей.",
    )
    parser.add_argument(
        "--normal-only",
        action="store_true",
        default=RAYS_NORMAL_ONLY,
        help="Экспортировать только луч нормали каждой точки.",
    )
    parser.add_argument(
        "--no-force-normal",
        action="store_false",
        dest="always_include_normal",
        default=RAYS_ALWAYS_INCLUDE_NORMAL,
        help="Не добавлять луч нормали, если он вне сетки шага.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    step = args.ray_step_deg
    if step is not None:
        if not math.isfinite(step) or step <= 0.0 or step > 360.0:
            raise ValueError(
                f"--ray-step-deg должен быть в диапазоне (0, 360], "
                f"получено {step}"
            )
    if not math.isfinite(args.ray_origin_deg):
        raise ValueError("--ray-origin-deg должен быть конечным числом")


def resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def setup_env(database_path: Path) -> None:
    if "COASTLINE_DATABASE_URL" not in os.environ:
        db_path = resolve_path(database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{db_path.as_posix()}"
        os.environ["COASTLINE_DATABASE_URL"] = db_url
        logger.debug(f"COASTLINE_DATABASE_URL -> {db_url}")


def _angular_distance(a: float, b: float) -> float:
    """Минимальный угол между двумя азимутами [0, 180]."""
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def _on_step_grid(
    azimuth_deg: float,
    step_deg: float,
    origin_deg: float,
    tolerance_deg: float,
) -> bool:
    """
    True, если азимут лежит на сетке origin + k * step.

    Сравнение циклическое: остаток берётся по модулю шага,
    поэтому 359.9999999 и 0.0 считаются одним узлом сетки.
    Для шага, не делящего 360° нацело, последний интервал короче шага.
    """
    remainder = (azimuth_deg - origin_deg) % step_deg
    return min(remainder, step_deg - remainder) <= tolerance_deg


def _azimuth_to_endpoint(
    lon: float,
    lat: float,
    azimuth_deg: float,
    length_m: float,
) -> tuple[float, float]:
    """
    Прямая геодезическая задача на эллипсоиде WGS-84.
    Azimuth: от севера по часовой стрелке (0 = N, 90 = E).
    """
    from pyproj import Geod

    end_lon, end_lat, _ = Geod(ellps="WGS84").fwd(
        lon, lat, azimuth_deg, length_m
    )
    return float(end_lon), float(end_lat)


def _closest_to_normal(fetches: list, normal_az: float):
    return min(
        fetches,
        key=lambda f: _angular_distance(f.azimuth_deg, normal_az),
    )


def _bounds_warning(
    available_azimuths: list[float],
    step_deg: float,
    origin_deg: float,
) -> None:
    """Предупреждает, если сетка шага плохо согласована с данными БД."""
    unique = sorted({round(a % 360.0, 6) for a in available_azimuths})
    if len(unique) < 2:
        return

    diffs = [b - a for a, b in zip(unique, unique[1:])]
    db_step = min(d for d in diffs if d > 1e-9)

    ratio = step_deg / db_step
    if abs(ratio - round(ratio)) > 1e-6:
        logger.warning(
            f"Шаг экспорта {step_deg}° не кратен шагу данных БД "
            f"(≈{db_step}°): часть узлов сетки не будет иметь лучей."
        )


# ------------------------------------------------------------------
# Основная логика
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate_args(args)
    setup_env(args.database)

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

    output_dir = resolve_path(args.output_dir)
    Base.metadata.create_all(database.engine)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Определяем normal_source_id
    # ------------------------------------------------------------------
    with database.SessionLocal() as session:
        if args.normal_source_id is None:
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
            ns = session.get(
                CoastlineNormalSourceModel, args.normal_source_id
            )

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

        # --------------------------------------------------------------
        # 2. Нормали: point_id → normal_azimuth_deg
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # 3. Точки береговой линии
        # --------------------------------------------------------------
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
        point_by_id: dict[int, CoastlinePointModel] = {
            r.id: r for r in point_rows
        }
        logger.info(f"Точек загружено: {len(point_by_id)}")

        # --------------------------------------------------------------
        # 4. Фетчи
        # --------------------------------------------------------------
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
    fetches_by_point: dict[int, list] = defaultdict(list)
    for f in fetch_rows:
        fetches_by_point[f.point_id].append(f)

    # ------------------------------------------------------------------
    # 6. Слой точек с агрегированными атрибутами (по всем фетчам БД)
    # ------------------------------------------------------------------
    point_records: list[dict] = []

    for pid, p in point_by_id.items():
        normal_az = normal_by_point.get(pid)
        fetches = fetches_by_point.get(pid, [])

        if fetches:
            lengths = [f.fetch_length_m for f in fetches]
            fetch_min = min(lengths)
            fetch_max = max(lengths)
            fetch_mean = round(sum(lengths) / len(lengths), 2)
            fetch_count = len(lengths)
            fetch_normal_m = (
                _closest_to_normal(fetches, normal_az).fetch_length_m
                if normal_az is not None
                else None
            )
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
                "fetch_mean_m":       fetch_mean,
                "fetch_normal_m":     fetch_normal_m,
                "geometry":           Point(p.lon, p.lat),
            }
        )

    points_gdf = gpd.GeoDataFrame(
        point_records, geometry="geometry", crs="EPSG:4326"
    )
    points_path = output_dir / "wind_fetch_points.geojson"
    points_gdf.to_file(str(points_path), driver="GeoJSON")
    logger.success(f"Экспортировано {len(points_gdf)} точек → {points_path}")

    # ------------------------------------------------------------------
    # 7. Слой лучей
    # ------------------------------------------------------------------
    if not args.export_rays:
        logger.info("Экспорт лучей пропущен (--no-rays)")
        return

    step = args.ray_step_deg
    origin = args.ray_origin_deg

    if step is None:
        logger.info("Шаг лучей: все лучи из БД")
    else:
        logger.info(
            f"Шаг лучей: {step}° от азимута {origin}° "
            f"(луч нормали всегда: {args.always_include_normal})"
        )
        _bounds_warning([f.azimuth_deg for f in fetch_rows], step, origin)

    ray_records: list[dict] = []
    skipped_zero_length = 0
    skipped_by_step = 0

    for pid, fetches in fetches_by_point.items():
        p = point_by_id.get(pid)
        if p is None:
            continue

        normal_az = normal_by_point.get(pid)

        # Луч, ближайший к нормали, определяется по ВСЕМ фетчам точки,
        # а не по отфильтрованным шагом.
        if normal_az is not None:
            normal_closest_az: float | None = _closest_to_normal(
                fetches, normal_az
            ).azimuth_deg
        else:
            normal_closest_az = None

        for f in fetches:
            is_normal = (
                normal_closest_az is not None
                and _angular_distance(f.azimuth_deg, normal_closest_az)
                <= NORMAL_RAY_TOLERANCE_DEG
            )

            if args.normal_only:
                if not is_normal:
                    continue
            elif step is not None:
                on_grid = _on_step_grid(
                    f.azimuth_deg,
                    step,
                    origin,
                    RAY_STEP_TOLERANCE_DEG,
                )
                keep_as_normal = args.always_include_normal and is_normal
                if not on_grid and not keep_as_normal:
                    skipped_by_step += 1
                    continue

            # Нулевая длина даёт вырожденный LineString
            if f.fetch_length_m is None or f.fetch_length_m <= 0.0:
                skipped_zero_length += 1
                continue

            end_lon, end_lat = _azimuth_to_endpoint(
                lon=p.lon,
                lat=p.lat,
                azimuth_deg=f.azimuth_deg,
                length_m=f.fetch_length_m,
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

    if skipped_by_step:
        logger.info(f"Пропущено шагом {step}°: {skipped_by_step} лучей")
    if skipped_zero_length:
        logger.warning(
            f"Пропущено лучей нулевой длины: {skipped_zero_length}"
        )

    if not ray_records:
        logger.warning(
            "Ни один луч не прошёл фильтрацию. "
            "Проверьте --ray-step-deg и --ray-origin-deg: "
            "азимуты в БД могут не попадать в заданную сетку."
        )
        return

    rays_gdf = gpd.GeoDataFrame(
        ray_records, geometry="geometry", crs="EPSG:4326"
    )
    rays_path = output_dir / "wind_fetch_rays.geojson"
    rays_gdf.to_file(str(rays_path), driver="GeoJSON")
    logger.success(f"Экспортировано {len(rays_gdf)} лучей → {rays_path}")


if __name__ == "__main__":
    main()
