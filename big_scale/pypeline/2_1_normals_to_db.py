"""
big_scale/pypelin/2_1_normals_to_db.py

Расчёт нормалей, проверка по маске суши и запись в SQLite.

По умолчанию:
- неоднозначные точки исключаются из расчёта;
- исключённые точки сохраняются в отдельный GeoJSON;
- затем удаляются из coastline_points;
- coastline_sources.points_count обновляется;
- подтверждённые нормали сохраняются в БД.

Удаление точек и запись нормалей выполняются одной транзакцией.

Защита:
- точки с существующими внешними ссылками не удаляются;
- при ошибке транзакция откатывается;
- повторное имя набора нормалей по умолчанию запрещено.

Файлы src/ не изменяются.

Ограничение:
существующая математика нормалей использует одну рабочую
метрическую проекцию. nx/ny и азимут относятся к её осям,
а не автоматически к истинному геодезическому северу.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
from loguru import logger
from pyproj import CRS
from shapely import STRtree, make_valid
from shapely.geometry import Point
from shapely.validation import explain_validity
from sqlalchemy import (
    create_engine,
    delete,
    event,
    insert,
    select,
)
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
from tqdm.auto import tqdm


# ==================== КОРЕНЬ ПРОЕКТА ====================

def find_project_root() -> Path:
    directory = Path(__file__).resolve().parent

    for candidate in (directory, *directory.parents):
        if (
            candidate / "src/coastline/storage/models.py"
        ).is_file():
            return candidate

    raise RuntimeError(
        "Не найден корень BreachTheBeach: "
        "ожидался src/coastline/storage/models.py."
    )


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.coastline.domain.CoastlineDataset import (  # noqa: E402
    CoastlineDataset,
)
from src.coastline.domain.CoastlineNormalPointSet import (  # noqa: E402
    CoastlineNormalPointSet,
)
from src.coastline.domain.CoastlinePointSet import (  # noqa: E402
    CoastlinePointSet,
    PointSetMeta,
)
from src.coastline.services.CoastlineNormalService import (  # noqa: E402
    CoastlineNormalConfig,
    CoastlineNormalService,
)
from src.coastline.storage.models import (  # noqa: E402
    Base,
    CoastlineNormalModel,
    CoastlineNormalSourceModel,
    CoastlinePointModel,
    CoastlineSourceModel,
)


# ==================== НАСТРОЙКИ ====================

DATABASE_PATH = Path(
    "big_scale/data/db/coastline.db"
)

POINT_SOURCE_ID: int | None = None

POINT_SOURCE_NAME: str | None = (
    "S2Coast_Russia_imported_points"
)

# None — использовать main_geojson_path из SQLite.
COASTLINE_MAIN_PATH: Path | None = None

LAND_POLYGONS_PATH = Path(
    "big_scale/data/S2Coast2023_ShapeFile_vector/"
    "S2Coast-2023_Polygon_fishnet.shp"
)

NORMALS_NAME = "S2Coast_normals_land_checked"

INITIAL_SEA_SIDE = "right"

TANGENT_DELTA_M = 10.0

MAX_REFERENCE_DISTANCE_M: float | None = 50.0

WORKING_CRS: str | None = None

LAND_CHECK_DISTANCES_M = (
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
)

# True — исключать неоднозначные точки.
# False — остановиться, если такие точки обнаружены.
DROP_AMBIGUOUS_POINTS = True

# True — также удалить исключённые точки из SQLite
# и обновить points_count источника.
# False — оставить исходные точки в БД, сохранив только
# подмножество подтверждённых нормалей.
DELETE_DROPPED_POINTS_FROM_DATABASE = True

DIAGNOSTIC_GEOJSON_PATH: Path | None = Path(
    "big_scale/data/coastline/normals_land_checked.geojson"
)

# Полный набор диагностических нормалей, включая неоднозначные.
# Содержит land_check_resolved и normal_flipped.
ALL_NORMALS_DIAGNOSTIC_PATH: Path | None = Path(
    "big_scale/data/coastline/normals_land_check_all.geojson"
)

# Точки, исключённые перед записью в БД.
REJECTED_POINTS_GEOJSON_PATH = Path(
    "big_scale/data/coastline/normals_rejected_points.geojson"
)

DIAGNOSTIC_NORMAL_LENGTH_M = 1000.0

SEARCH_BATCH_SIZE = 2_000
LAND_CHECK_BATCH_SIZE = 10_000
INSERT_BATCH_SIZE = 10_000

# Для SQL-запросов с IN, чтобы не превышать лимит параметров SQLite.
SQL_ID_BATCH_SIZE = 500

ALLOW_DUPLICATE_NORMALS_NAME = False

RESTORE_MISSING_SHX = True


# ==================== ПУТИ ====================

def resolve_path(path: str | Path) -> Path:
    result = Path(path).expanduser()

    if not result.is_absolute():
        result = PROJECT_ROOT / result

    return result.resolve()


def require_file(
    path: str | Path,
    description: str,
) -> Path:
    result = resolve_path(path)

    if not result.is_file():
        raise FileNotFoundError(
            f"{description} не найден:\n{result}"
        )

    return result


# ==================== ПРОВЕРКА НАСТРОЕК ====================

def validate_settings() -> None:
    if INITIAL_SEA_SIDE not in {"left", "right"}:
        raise ValueError(
            "INITIAL_SEA_SIDE должен быть left или right."
        )

    if not NORMALS_NAME.strip():
        raise ValueError("NORMALS_NAME не должен быть пустым.")

    if (
        not math.isfinite(TANGENT_DELTA_M)
        or TANGENT_DELTA_M <= 0
    ):
        raise ValueError("TANGENT_DELTA_M должен быть > 0.")

    if MAX_REFERENCE_DISTANCE_M is not None:
        if (
            not math.isfinite(MAX_REFERENCE_DISTANCE_M)
            or MAX_REFERENCE_DISTANCE_M < 0
        ):
            raise ValueError(
                "MAX_REFERENCE_DISTANCE_M должен быть >= 0 или None."
            )

    if not LAND_CHECK_DISTANCES_M or any(
        not math.isfinite(value) or value <= 0
        for value in LAND_CHECK_DISTANCES_M
    ):
        raise ValueError(
            "Расстояния проверки суши должны быть "
            "положительными и конечными."
        )

    for name, value in (
        ("SEARCH_BATCH_SIZE", SEARCH_BATCH_SIZE),
        ("LAND_CHECK_BATCH_SIZE", LAND_CHECK_BATCH_SIZE),
        ("INSERT_BATCH_SIZE", INSERT_BATCH_SIZE),
        ("SQL_ID_BATCH_SIZE", SQL_ID_BATCH_SIZE),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{name} должен быть положительным целым."
            )

    if (
        not math.isfinite(DIAGNOSTIC_NORMAL_LENGTH_M)
        or DIAGNOSTIC_NORMAL_LENGTH_M <= 0
    ):
        raise ValueError(
            "DIAGNOSTIC_NORMAL_LENGTH_M должен быть > 0."
        )


# ==================== ЛОКАЛЬНЫЙ АДАПТЕР ====================

class PipelineNormalService(CoastlineNormalService):
    """
    Базовый сервис не изменяется.

    Для открытых линий сначала используется его алгоритм.
    Замкнутые линии обрабатываются циклическим окном.
    """

    def __init__(self, config):
        super().__init__(config)

        self.closed_tangent_count = 0
        self.fallback_tangent_count = 0
        self.edge_fallback_count = 0

    def _tangent_points(self, coastline, chainage):
        length = float(coastline.length)

        if not math.isfinite(length) or length <= 0:
            raise ValueError(
                f"Некорректная длина линии: {length!r}"
            )

        if not math.isfinite(chainage):
            raise ValueError(
                f"Некорректный chainage: {chainage!r}"
            )

        requested = float(self.config.tangent_delta_m)
        closed = bool(coastline.is_closed)

        if not closed:
            try:
                return super()._tangent_points(
                    coastline,
                    chainage,
                )
            except ValueError:
                self.fallback_tangent_count += 1
        else:
            self.closed_tangent_count += 1

        coordinates = list(coastline.coords)

        if not all(
            math.isfinite(value)
            for coordinate in coordinates
            for value in coordinate[:2]
        ):
            raise ValueError(
                "Линия содержит неконечные координаты."
            )

        scale = max(
            1.0,
            max(
                abs(value)
                for coordinate in coordinates
                for value in coordinate[:2]
            ),
        )

        tolerance = max(
            1e-9,
            8.0 * math.ulp(scale),
        )

        center_s = (
            chainage % length
            if closed
            else min(max(chainage, 0.0), length)
        )

        def point_at(distance):
            if closed:
                distance %= length
            else:
                distance = min(max(distance, 0.0), length)

            return coastline.interpolate(distance)

        initial_delta = min(
            requested,
            length / 4.0 if closed else length,
        )

        delta = initial_delta

        for _ in range(32):
            p0 = point_at(center_s - delta)
            p1 = point_at(center_s + delta)

            if p0.distance(p1) > tolerance:
                return p0, p1, delta

            delta *= 0.5

        center = point_at(center_s)
        delta = initial_delta

        for _ in range(32):
            forward = point_at(center_s + delta)

            if center.distance(forward) > tolerance:
                return center, forward, delta

            backward = point_at(center_s - delta)

            if backward.distance(center) > tolerance:
                return backward, center, delta

            delta *= 0.5

        best = None
        best_score = math.inf
        accumulated = 0.0

        for c0, c1 in zip(coordinates, coordinates[1:]):
            p0 = Point(c0[:2])
            p1 = Point(c1[:2])

            edge_length = float(p0.distance(p1))

            edge_start = accumulated
            edge_end = edge_start + edge_length
            accumulated = edge_end

            if edge_length <= tolerance:
                continue

            positions = (
                (
                    center_s - length,
                    center_s,
                    center_s + length,
                )
                if closed
                else (center_s,)
            )

            score = min(
                max(
                    edge_start - position,
                    position - edge_end,
                    0.0,
                )
                for position in positions
            )

            if score < best_score:
                best_score = score

                best = (
                    p0,
                    p1,
                    min(requested, edge_length),
                )

        if best is None:
            raise ValueError(
                "Нет устойчивого ненулевого ребра "
                "для вычисления касательной."
            )

        self.edge_fallback_count += 1

        return best


# ==================== РАСЧЁТ НОРМАЛЕЙ ====================

def calculate_normals(point_set, dataset, config):
    service = PipelineNormalService(config)

    logger.info(
        "Подготовка береговых компонентов и рабочей CRS..."
    )

    parts, work_crs = service._prepare_main_lines(dataset)

    points = service._prepare_points(
        point_set,
        work_crs,
    ).reset_index(drop=True)

    crs = CRS.from_user_input(work_crs)

    if not crs.is_projected or any(
        not math.isclose(
            axis.unit_conversion_factor,
            1.0,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        for axis in crs.axis_info[:2]
    ):
        raise ValueError(
            "Рабочая CRS должна быть проекционной "
            "с единицами в метрах."
        )

    logger.info(
        "Построение STRtree: {} компонентов",
        len(parts),
    )

    tree = STRtree(parts)

    geometries = points.geometry.to_numpy()

    attributes = points.drop(
        columns=[points.geometry.name]
    ).to_dict(orient="records")

    count = len(points)

    part_ids = np.full(count, -1, dtype=np.int64)
    distances = np.full(count, np.nan, dtype=float)

    with tqdm(
        total=count,
        desc="Поиск ближайшего берега",
        unit="точ.",
        dynamic_ncols=True,
    ) as progress:
        for start in range(0, count, SEARCH_BATCH_SIZE):
            stop = min(start + SEARCH_BATCH_SIZE, count)

            pairs, batch_distances = tree.query_nearest(
                geometries[start:stop],
                all_matches=True,
                return_distance=True,
            )

            if pairs.size:
                order = np.lexsort(
                    (
                        pairs[1],
                        batch_distances,
                        pairs[0],
                    )
                )

                sorted_points = pairs[0, order]

                first = np.r_[
                    True,
                    sorted_points[1:] != sorted_points[:-1],
                ]

                selected = order[first]
                positions = start + pairs[0, selected]

                part_ids[positions] = pairs[1, selected]
                distances[positions] = batch_distances[selected]

            progress.update(stop - start)

    if (
        (part_ids < 0).any()
        or not np.isfinite(distances).all()
    ):
        raise RuntimeError(
            "Не для всех точек найден ближайший берег."
        )

    maximum = config.max_reference_distance_m

    if maximum is not None:
        bad = np.flatnonzero(distances > maximum)

        if len(bad):
            examples = [
                {
                    "point_id": int(
                        attributes[int(i)]["point_id"]
                    ),
                    "distance_m": float(distances[i]),
                }
                for i in bad[:10]
            ]

            raise ValueError(
                f"Слишком далеко от берега: {len(bad)} точек. "
                f"Допуск: {maximum} м. Примеры: {examples}"
            )

    logger.info(
        "Максимальное расстояние до берега: {:.3f} м",
        float(distances.max()),
    )

    records = []

    for i in tqdm(
        range(count),
        desc="Расчёт нормалей",
        unit="точ.",
        dynamic_ncols=True,
    ):
        attrs = attributes[i]
        part_id = int(part_ids[i])

        try:
            info = service._compute_normal_at_point(
                geometries[i],
                parts[part_id],
            )

        except (ValueError, RuntimeError) as error:
            raise ValueError(
                f"point_id={attrs['point_id']}, "
                f"seq={attrs['seq']}, "
                f"part_id={part_id}: {error}"
            ) from error

        records.append(
            {
                **attrs,
                **info,
                "reference_part_id": part_id,
                "reference_distance_m": float(distances[i]),
                "sea_side": config.sea_side,
                "geometry": geometries[i],
            }
        )

    logger.info(
        "Касательные: замкнутых={}, резервных открытых={}, "
        "по отдельному ребру={}",
        service.closed_tangent_count,
        service.fallback_tangent_count,
        service.edge_fallback_count,
    )

    return CoastlineNormalPointSet.from_gdf(
        gpd.GeoDataFrame(
            records,
            geometry="geometry",
            crs=work_crs,
        ),
        name=NORMALS_NAME,
    )


# ==================== ИСПРАВЛЕНИЕ МАСКИ СУШИ ====================

def iter_polygon_parts(geometry):
    if geometry is None or geometry.is_empty:
        return

    if geometry.geom_type == "Polygon":
        yield geometry

    elif geometry.geom_type in {
        "MultiPolygon",
        "GeometryCollection",
    }:
        for part in geometry.geoms:
            yield from iter_polygon_parts(part)


def repair_land_geometries(land, stage):
    polygons = []
    lost_indices = []
    repaired_count = 0

    for index, geometry in tqdm(
        land.geometry.items(),
        total=len(land),
        desc=f"Подготовка суши: {stage}",
        unit="об.",
        dynamic_ncols=True,
    ):
        if geometry is None or geometry.is_empty:
            lost_indices.append(index)
            continue

        if not geometry.is_valid:
            logger.warning(
                "{}: объект {}, причина: {}",
                stage,
                index,
                explain_validity(geometry),
            )

            geometry = make_valid(geometry)
            repaired_count += 1

        parts = [
            polygon
            for polygon in iter_polygon_parts(geometry)
            if not polygon.is_empty and polygon.area > 0
        ]

        if not parts:
            lost_indices.append(index)
            continue

        if not all(part.is_valid for part in parts):
            raise ValueError(
                f"Не удалось исправить полигон {index}."
            )

        polygons.extend(parts)

    if lost_indices:
        raise ValueError(
            "После исправления потеряна полигональная часть "
            f"у {len(lost_indices)} объектов. "
            f"Первые индексы: {lost_indices[:20]}"
        )

    if not polygons:
        raise ValueError("После обработки маска суши пуста.")

    logger.info(
        "{}: исправлено объектов — {}; "
        "полигональных частей — {}",
        stage,
        repaired_count,
        len(polygons),
    )

    return gpd.GeoDataFrame(
        geometry=polygons,
        crs=land.crs,
    )


def load_land_polygons(path):
    logger.info("Чтение полигонов суши: {}", path)

    if RESTORE_MISSING_SHX and path.suffix.lower() == ".shp":
        pyogrio.set_gdal_config_options(
            {"SHAPE_RESTORE_SHX": "YES"}
        )

    land = gpd.read_file(
        path,
        engine="pyogrio",
        columns=[],
    )

    if land.crs is None:
        raise ValueError("У маски суши отсутствует CRS.")

    nonempty = (
        land.geometry.notna()
        & ~land.geometry.is_empty
    )

    empty_count = int((~nonempty).sum())

    if empty_count:
        logger.warning(
            "Пропущено пустых геометрий: {}",
            empty_count,
        )

    land = land.loc[nonempty].copy()

    if land.empty:
        raise ValueError("Маска суши пуста.")

    if not land.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    ).all():
        raise TypeError(
            "Ожидались Polygon/MultiPolygon."
        )

    land = repair_land_geometries(
        land,
        stage="исходная CRS",
    )

    target = CRS.from_epsg(4326)

    if not CRS.from_user_input(land.crs).equals(
        target,
        ignore_axis_order=True,
    ):
        land = land.to_crs(target)

        if not land.geometry.is_valid.all():
            land = repair_land_geometries(
                land,
                stage="WGS84",
            )
    else:
        land = land.set_crs(
            target,
            allow_override=True,
        )

    land = land.reset_index(drop=True)

    if (
        land.empty
        or land.geometry.isna().any()
        or land.geometry.is_empty.any()
        or not land.geometry.is_valid.all()
        or not np.isfinite(land.total_bounds).all()
    ):
        raise ValueError(
            "Маска суши не прошла итоговую проверку."
        )

    logger.info(
        "Построение индекса суши: {} полигонов",
        len(land),
    )

    _ = land.sindex

    return land


def points_on_land(points, land):
    geographic = points.to_crs(land.crs)

    lon = geographic.x.to_numpy()
    lat = geographic.y.to_numpy()

    if not (
        np.isfinite(lon).all()
        and np.isfinite(lat).all()
    ):
        raise ValueError(
            "Некорректные координаты контрольных точек."
        )

    lon = ((lon + 180.0) % 360.0) - 180.0

    query_points = gpd.points_from_xy(
        lon,
        lat,
        crs=land.crs,
    )

    hits = land.sindex.query(
        query_points,
        predicate="intersects",
    )

    result = np.zeros(len(points), dtype=bool)

    if hits.size:
        result[np.unique(hits[0])] = True

    meridian = np.flatnonzero(lon == -180.0)

    if len(meridian):
        alternative = gpd.points_from_xy(
            np.full(len(meridian), 180.0),
            lat[meridian],
            crs=land.crs,
        )

        hits = land.sindex.query(
            alternative,
            predicate="intersects",
        )

        if hits.size:
            result[
                meridian[np.unique(hits[0])]
            ] = True

    return result


# ==================== ПРОВЕРКА НАПРАВЛЕНИЙ ====================

def orient_normals_to_water(normal_set, land):
    gdf = normal_set.gdf.reset_index(drop=True).copy()

    x = gdf.geometry.x.to_numpy(dtype=float)
    y = gdf.geometry.y.to_numpy(dtype=float)
    nx = gdf["nx"].to_numpy(dtype=float).copy()
    ny = gdf["ny"].to_numpy(dtype=float).copy()

    resolved = np.zeros(len(gdf), dtype=bool)
    flipped = np.zeros(len(gdf), dtype=bool)
    check_distance = np.full(len(gdf), np.nan)

    for distance in sorted(set(LAND_CHECK_DISTANCES_M)):
        unresolved = np.flatnonzero(~resolved)

        if not len(unresolved):
            break

        with tqdm(
            total=len(unresolved),
            desc=f"Проверка суши, {distance:g} м",
            unit="точ.",
            dynamic_ncols=True,
        ) as progress:
            for start in range(
                0,
                len(unresolved),
                LAND_CHECK_BATCH_SIZE,
            ):
                ids = unresolved[
                    start:start + LAND_CHECK_BATCH_SIZE
                ]

                forward = gpd.GeoSeries(
                    gpd.points_from_xy(
                        x[ids] + nx[ids] * distance,
                        y[ids] + ny[ids] * distance,
                    ),
                    crs=gdf.crs,
                )

                backward = gpd.GeoSeries(
                    gpd.points_from_xy(
                        x[ids] - nx[ids] * distance,
                        y[ids] - ny[ids] * distance,
                    ),
                    crs=gdf.crs,
                )

                forward_land = points_on_land(forward, land)
                backward_land = points_on_land(backward, land)

                clear = forward_land != backward_land
                clear_ids = ids[clear]

                resolved[clear_ids] = True
                check_distance[clear_ids] = distance

                flipped[
                    ids[forward_land & ~backward_land]
                ] = True

                progress.update(len(ids))

        logger.info(
            "Расстояние {} м: определено {}, "
            "разворотов {}, осталось {}",
            distance,
            int(resolved.sum()),
            int(flipped.sum()),
            int((~resolved).sum()),
        )

    nx[flipped] *= -1.0
    ny[flipped] *= -1.0

    gdf["nx"] = nx
    gdf["ny"] = ny

    gdf["normal_azimuth_deg"] = (
        np.degrees(np.arctan2(nx, ny)) + 360.0
    ) % 360.0

    gdf.loc[flipped, "sea_side"] = (
        gdf.loc[flipped, "sea_side"].map(
            {"left": "right", "right": "left"}
        )
    )

    gdf["normal_flipped"] = flipped
    gdf["land_check_resolved"] = resolved
    gdf["land_check_distance_m"] = check_distance

    normal_set.gdf = gdf

    return int(flipped.sum())


# ==================== ДИАГНОСТИКА ====================

def export_points_geojson(gdf, path):
    """
    Сохраняет точки, включая пустой результат.
    Запись во временный файл с последующей заменой.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    try:
        geographic = gdf.to_crs("EPSG:4326")

        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                '{"type":"FeatureCollection","features":[\n'
            )

            first = True

            for feature in geographic.iterfeatures(
                na="null",
                drop_id=True,
            ):
                if not first:
                    stream.write(",\n")

                json.dump(
                    feature,
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )

                first = False

            stream.write("\n]}\n")

        temporary.replace(path)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def export_normal_lines(normal_set, path):
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Сохранение диагностических нормалей: {}",
        path,
    )

    normal_set.export_normal_lines_geojson(
        path=path,
        normal_length_m=DIAGNOSTIC_NORMAL_LENGTH_M,
    )


# ==================== ПРОВЕРКИ БАЗЫ ====================

def check_existing_normal_source(session, point_source_id):
    existing_id = session.scalar(
        select(CoastlineNormalSourceModel.id)
        .where(
            CoastlineNormalSourceModel.point_source_id
            == point_source_id,
            CoastlineNormalSourceModel.name == NORMALS_NAME,
        )
        .limit(1)
    )

    if (
        existing_id is not None
        and not ALLOW_DUPLICATE_NORMALS_NAME
    ):
        raise ValueError(
            f"Набор {NORMALS_NAME!r} уже существует: "
            f"id={existing_id}. Измените NORMALS_NAME."
        )


def quote_identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def ensure_no_point_dependents(session, point_ids):
    """
    Проверяет все объявленные внешние ключи SQLite,
    ссылающиеся на coastline_points.id.

    Не допускает неявного каскадного удаления результатов.
    """
    if not point_ids:
        return

    connection = session.connection()

    tables = connection.exec_driver_sql(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        """
    ).scalars().all()

    for table in tables:
        foreign_keys = connection.exec_driver_sql(
            f"PRAGMA foreign_key_list({quote_identifier(table)})"
        ).mappings().all()

        for foreign_key in foreign_keys:
            if foreign_key["table"] != "coastline_points":
                continue

            target_column = foreign_key["to"]

            if target_column not in (None, "", "id"):
                raise RuntimeError(
                    "Обнаружена нестандартная внешняя ссылка "
                    f"на coastline_points: таблица {table!r}. "
                    "Удаление остановлено."
                )

            column = foreign_key["from"]

            for start in range(
                0,
                len(point_ids),
                SQL_ID_BATCH_SIZE,
            ):
                batch = point_ids[
                    start:start + SQL_ID_BATCH_SIZE
                ]

                placeholders = ",".join("?" for _ in batch)

                query = (
                    f"SELECT 1 FROM {quote_identifier(table)} "
                    f"WHERE {quote_identifier(column)} "
                    f"IN ({placeholders}) LIMIT 1"
                )

                found = connection.exec_driver_sql(
                    query,
                    tuple(batch),
                ).first()

                if found is not None:
                    raise RuntimeError(
                        "Нельзя безопасно удалить неоднозначные "
                        "точки: на них уже ссылаются записи "
                        f"{table}.{column}. "
                        "Связанные результаты не удалены. "
                        "Можно использовать новую базу либо "
                        "отключить физическое удаление через "
                        "DELETE_DROPPED_POINTS_FROM_DATABASE=False."
                    )


# ==================== ОСНОВНОЙ ПАЙПЛАЙН ====================

def main():
    validate_settings()

    database_path = require_file(
        DATABASE_PATH,
        "База данных",
    )

    land_path = require_file(
        LAND_POLYGONS_PATH,
        "Маска суши",
    )

    engine = create_engine(
        URL.create(
            "sqlite",
            database=str(database_path),
        ),
        connect_args={"timeout": 60},
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, connection_record):
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
        finally:
            cursor.close()

    try:
        logger.info("База: {}", database_path)

        Base.metadata.create_all(engine)

        with Session(engine) as session:
            if POINT_SOURCE_ID is not None:
                source = session.get(
                    CoastlineSourceModel,
                    POINT_SOURCE_ID,
                )
            else:
                query = select(CoastlineSourceModel)

                if POINT_SOURCE_NAME is not None:
                    query = query.where(
                        CoastlineSourceModel.name
                        == POINT_SOURCE_NAME
                    )

                source = session.scalars(
                    query.order_by(
                        CoastlineSourceModel.id.desc()
                    ).limit(1)
                ).first()

            if source is None:
                raise ValueError(
                    "Источник точек не найден."
                )

            source_id = int(source.id)
            source_name = str(source.name)
            source_crs = source.crs
            metadata_main_path = source.main_geojson_path
            expected_count = int(source.points_count)

            check_existing_normal_source(
                session,
                source_id,
            )

            rows = session.execute(
                select(
                    CoastlinePointModel.id,
                    CoastlinePointModel.seq,
                    CoastlinePointModel.lon,
                    CoastlinePointModel.lat,
                )
                .where(
                    CoastlinePointModel.source_id == source_id
                )
                .order_by(
                    CoastlinePointModel.seq,
                    CoastlinePointModel.id,
                )
            ).all()

        if not rows or len(rows) != expected_count:
            raise ValueError(
                "Нет точек либо количество не соответствует "
                "метаданным источника."
            )

        if source_crs is None or not CRS.from_user_input(
            source_crs
        ).equals(
            CRS.from_epsg(4326),
            ignore_axis_order=True,
        ):
            raise ValueError(
                "Точки базы должны быть в EPSG:4326."
            )

        original_ids = [int(row.id) for row in rows]
        seqs = [int(row.seq) for row in rows]

        if len(seqs) != len(set(seqs)):
            raise ValueError("Повторяющиеся seq.")

        xy = np.array(
            [
                [float(row.lon), float(row.lat)]
                for row in rows
            ],
            dtype=float,
        )

        if not (
            np.isfinite(xy).all()
            and np.all(np.abs(xy[:, 0]) <= 180)
            and np.all(np.abs(xy[:, 1]) <= 90)
        ):
            raise ValueError(
                "Некорректные координаты в базе."
            )

        reference_value = (
            COASTLINE_MAIN_PATH
            if COASTLINE_MAIN_PATH is not None
            else metadata_main_path
        )

        if not reference_value:
            raise ValueError(
                "Не указан путь к береговой линии."
            )

        reference_path = require_file(
            reference_value,
            "Береговая линия",
        )

        diagnostic_path = (
            resolve_path(DIAGNOSTIC_GEOJSON_PATH)
            if DIAGNOSTIC_GEOJSON_PATH is not None
            else None
        )

        all_normals_path = (
            resolve_path(ALL_NORMALS_DIAGNOSTIC_PATH)
            if ALL_NORMALS_DIAGNOSTIC_PATH is not None
            else None
        )

        rejected_path = resolve_path(
            REJECTED_POINTS_GEOJSON_PATH
        )

        output_paths = [
            path
            for path in (
                diagnostic_path,
                all_normals_path,
                rejected_path,
            )
            if path is not None
        ]

        if len(output_paths) != len(set(output_paths)):
            raise ValueError(
                "Выходные диагностические пути должны различаться."
            )

        protected_paths = {
            database_path,
            land_path,
            reference_path,
        }

        if any(path in protected_paths for path in output_paths):
            raise ValueError(
                "Выходной файл совпадает с входным."
            )

        if rejected_path.with_name(
            rejected_path.name + ".tmp"
        ) in protected_paths:
            raise ValueError(
                "Временный файл совпадает с входным."
            )

        logger.info(
            "Источник: id={}, name={!r}, points={}",
            source_id,
            source_name,
            len(rows),
        )

        logger.info(
            "Береговая линия: {}",
            reference_path,
        )

        # Проверяем маску до расчёта нормалей.
        land = load_land_polygons(land_path)

        points_gdf = gpd.GeoDataFrame(
            {
                "point_id": original_ids,
                "seq": seqs,
            },
            geometry=gpd.points_from_xy(
                xy[:, 0],
                xy[:, 1],
            ),
            crs="EPSG:4326",
        )

        point_set = CoastlinePointSet(
            gdf=points_gdf,
            meta=PointSetMeta(
                name=NORMALS_NAME,
                source_dataset_name=source_name,
                strategy_name="database",
                source_mode="main",
                points_count=len(points_gdf),
            ),
        )

        dataset = CoastlineDataset.from_geojson(
            main_path=reference_path,
            other_path=None,
            name=NORMALS_NAME,
        )

        config = CoastlineNormalConfig(
            sea_side=INITIAL_SEA_SIDE,
            tangent_delta_m=TANGENT_DELTA_M,
            working_crs=(
                WORKING_CRS
                if WORKING_CRS is not None
                else str(dataset.metric_crs)
            ),
            max_reference_distance_m=MAX_REFERENCE_DISTANCE_M,
        )

        normal_set = calculate_normals(
            point_set,
            dataset,
            config,
        )

        calculated_ids = (
            normal_set.gdf["point_id"]
            .astype(int)
            .tolist()
        )

        if (
            len(calculated_ids) != len(original_ids)
            or len(set(calculated_ids)) != len(calculated_ids)
            or set(calculated_ids) != set(original_ids)
        ):
            raise RuntimeError(
                "Нормали не соответствуют исходным точкам."
            )

        if not normal_set.validate_vectors().is_valid:
            raise RuntimeError(
                normal_set.debug_report(n=10)
            )

        flipped_count = orient_normals_to_water(
            normal_set,
            land,
        )

        if not normal_set.validate_vectors().is_valid:
            raise RuntimeError(
                "Ошибка векторов после проверки суши:\n"
                + normal_set.debug_report(n=10)
            )

        rejected = normal_set.gdf.loc[
            ~normal_set.gdf["land_check_resolved"]
        ].copy()

        rejected["rejection_reason"] = (
            "ambiguous_land_water_direction"
        )

        dropped_ids = (
            rejected["point_id"].astype(int).tolist()
        )

        # Сохраняем исходные точки SQLite, а не их координаты
        # после прямого/обратного преобразования рабочей CRS.
        rejected_points = points_gdf.loc[
            points_gdf["point_id"].isin(dropped_ids)
        ].copy()

        rejected_points["source_id"] = source_id
        rejected_points["rejection_reason"] = (
            "ambiguous_land_water_direction"
        )

        export_points_geojson(
            rejected_points,
            rejected_path,
        )

        logger.info(
            "Неоднозначных точек: {}. Список: {}",
            len(dropped_ids),
            rejected_path,
        )

        export_normal_lines(
            normal_set,
            all_normals_path,
        )

        if dropped_ids and not DROP_AMBIGUOUS_POINTS:
            raise RuntimeError(
                f"Обнаружено неоднозначных нормалей: "
                f"{len(dropped_ids)}. "
                "Включите DROP_AMBIGUOUS_POINTS=True "
                "для их исключения. БД не изменена."
            )

        accepted = normal_set.gdf.loc[
            normal_set.gdf["land_check_resolved"]
        ].copy()

        if accepted.empty:
            raise RuntimeError(
                "Не осталось подтверждённых нормалей. "
                "БД не изменена."
            )

        # Исходные point_id и seq сохраняются:
        # пропуски в seq после удаления допустимы.
        normal_set = CoastlineNormalPointSet.from_gdf(
            accepted.reset_index(drop=True),
            name=NORMALS_NAME,
        )

        if not normal_set.validate_vectors().is_valid:
            raise RuntimeError(
                "Ошибка проверки отфильтрованных нормалей:\n"
                + normal_set.debug_report(n=10)
            )

        accepted_ids = set(
            normal_set.gdf["point_id"].astype(int)
        )

        if accepted_ids & set(dropped_ids):
            raise RuntimeError(
                "Принятые и исключённые point_id пересекаются."
            )

        if accepted_ids | set(dropped_ids) != set(original_ids):
            raise RuntimeError(
                "Потеряна согласованность идентификаторов."
            )

        export_normal_lines(
            normal_set,
            diagnostic_path,
        )

        sides = set(normal_set.gdf["sea_side"])

        stored_side = (
            next(iter(sides))
            if len(sides) == 1
            else "mixed"
        )

        values = normal_set.gdf[
            ["point_id", "nx", "ny", "normal_azimuth_deg"]
        ]

        actually_deleted = 0

        # Одна транзакция: удаление + обновление источника
        # + сохранение подтверждённых нормалей.
        with Session(engine) as session:
            with session.begin():
                # Запрещаем параллельному писателю изменить
                # набор точек между проверкой и удалением.
                session.connection().exec_driver_sql(
                    "BEGIN IMMEDIATE"
                )

                check_existing_normal_source(
                    session,
                    source_id,
                )

                current_rows = session.execute(
                    select(
                        CoastlinePointModel.id,
                        CoastlinePointModel.seq,
                        CoastlinePointModel.lon,
                        CoastlinePointModel.lat,
                    )
                    .where(
                        CoastlinePointModel.source_id == source_id
                    )
                    .order_by(
                        CoastlinePointModel.seq,
                        CoastlinePointModel.id,
                    )
                ).all()

                original_snapshot = [
                    (
                        int(row.id),
                        int(row.seq),
                        float(row.lon),
                        float(row.lat),
                    )
                    for row in rows
                ]

                current_snapshot = [
                    (
                        int(row.id),
                        int(row.seq),
                        float(row.lon),
                        float(row.lat),
                    )
                    for row in current_rows
                ]

                if current_snapshot != original_snapshot:
                    raise RuntimeError(
                        "Исходные точки изменились во время "
                        "расчёта. Повторите запуск."
                    )

                source = session.get(
                    CoastlineSourceModel,
                    source_id,
                )

                if source is None:
                    raise RuntimeError(
                        "Источник удалён во время расчёта."
                    )

                if (
                    dropped_ids
                    and DELETE_DROPPED_POINTS_FROM_DATABASE
                ):
                    ensure_no_point_dependents(
                        session,
                        dropped_ids,
                    )

                    with tqdm(
                        total=len(dropped_ids),
                        desc="Удаление неоднозначных точек",
                        unit="точ.",
                        dynamic_ncols=True,
                    ) as progress:
                        for start in range(
                            0,
                            len(dropped_ids),
                            SQL_ID_BATCH_SIZE,
                        ):
                            batch = dropped_ids[
                                start:start + SQL_ID_BATCH_SIZE
                            ]

                            session.execute(
                                delete(CoastlinePointModel)
                                .where(
                                    CoastlinePointModel.source_id
                                    == source_id,
                                    CoastlinePointModel.id.in_(
                                        batch
                                    ),
                                ),
                                execution_options={
                                    "synchronize_session": False
                                },
                            )

                            actually_deleted += len(batch)
                            progress.update(len(batch))

                    remaining_ids = set(
                        session.scalars(
                            select(CoastlinePointModel.id)
                            .where(
                                CoastlinePointModel.source_id
                                == source_id
                            )
                        ).all()
                    )

                    if remaining_ids != accepted_ids:
                        raise RuntimeError(
                            "После удаления набор точек "
                            "не соответствует нормалям."
                        )

                    source.points_count = len(remaining_ids)

                normal_source = CoastlineNormalSourceModel(
                    name=NORMALS_NAME,
                    point_source_id=source_id,
                    sea_side=stored_side,
                    tangent_delta_m=TANGENT_DELTA_M,
                    working_crs=str(normal_set.gdf.crs),
                    result_crs=str(normal_set.gdf.crs),
                    normals_count=len(values),
                )

                session.add(normal_source)
                session.flush()

                saved_normal_source_id = int(
                    normal_source.id
                )

                with tqdm(
                    total=len(values),
                    desc="Запись нормалей",
                    unit="норм.",
                    dynamic_ncols=True,
                ) as progress:
                    for start in range(
                        0,
                        len(values),
                        INSERT_BATCH_SIZE,
                    ):
                        batch = [
                            {
                                "normal_source_id": (
                                    saved_normal_source_id
                                ),
                                "point_id": int(row.point_id),
                                "nx": float(row.nx),
                                "ny": float(row.ny),
                                "normal_azimuth_deg": float(
                                    row.normal_azimuth_deg
                                ),
                            }
                            for row in values.iloc[
                                start:start + INSERT_BATCH_SIZE
                            ].itertuples(index=False)
                        ]

                        session.execute(
                            insert(CoastlineNormalModel),
                            batch,
                        )

                        progress.update(len(batch))

                logger.info(
                    "Фиксация изменений SQLite..."
                )

        logger.success(
            "Готово: исходных точек={}, "
            "исключено={}, удалено из БД={}, "
            "сохранено нормалей={}, normal_source_id={}",
            len(original_ids),
            len(dropped_ids),
            actually_deleted,
            len(values),
            saved_normal_source_id,
        )

        logger.info(
            "Развёрнутых направлений до фильтрации: {}",
            flipped_count,
        )

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()