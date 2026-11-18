"""
Импорт готовых береговых точек из GeoJSON в coastline.db.

Рекомендуемое расположение:
    db_pypeline/1_1_import_coastline_points.py

Контракт хранения:
- coastline_points.lon и coastline_points.lat всегда EPSG:4326;
- coastline_sources.crs всегда EPSG:4326;
- main_geojson_path содержит путь к файлу БЕРЕГОВОЙ ЛИНИИ;
- путь к исходному файлу ТОЧЕК хранится в strategy_params;
- seq назначается по порядку объектов в GeoJSON, начиная с нуля;
- id точек назначается базой данных автоматически;
- относительные пути разрешаются относительно корня проекта.

Поведение при повторном запуске:
- существующие данные не удаляются;
- если источник с DATASET_NAME уже существует, импорт по умолчанию
  останавливается;
- для добавления ещё одного источника с тем же именем установите
  ALLOW_DUPLICATE_DATASET_NAME = True.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from loguru import logger
from sqlalchemy import create_engine, event, insert, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session


# ==================== КОРЕНЬ ПРОЕКТА ====================

# Позволяет расположить скрипт в db_pypeline/, utils/
# или другом подкаталоге репозитория.
def find_project_root() -> Path:
    script_directory = Path(__file__).resolve().parent

    for directory in (script_directory, *script_directory.parents):
        if (
            directory / "src" / "coastline" / "storage" / "models.py"
        ).is_file():
            return directory

    raise RuntimeError(
        "Не найден корень BreachTheBeach: "
        "ожидался src/coastline/storage/models.py."
    )


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.coastline.storage.models import (  # noqa: E402
    Base,
    CoastlinePointModel,
    CoastlineSourceModel,
)


# ==================== НАСТРОЙКИ ====================

# По умолчанию совпадает с расположением базы
# в db_pypeline/2_1_normals_to_db.py.
DATABASE_PATH = Path(
    "/Users/mihailvystrcil/Documents/My Programms/BreachTheBeach/big_scale/data/db/coastline.db"
)

# Готовые точки: Point GeoJSON.
POINTS_GEOJSON_PATH = Path(
    "/Users/mihailvystrcil/Documents/My Programms/BreachTheBeach/big_scale/data/S2CoastRussiaCropped_points_weather_covered_unique_min_distance_400m.geojson"
)

# Береговая линия: LineString / MultiLineString GeoJSON.
# Этот путь будет сохранён в метаданных для расчёта нормалей.
COASTLINE_MAIN_PATH = Path(
    "/Users/mihailvystrcil/Documents/My Programms/BreachTheBeach/big_scale/data/S2CoastRussiaCropped.geojson"
)

# Дополнительные береговые линии, если нужны.
COASTLINE_OTHER_PATH: Path | None = None

DATASET_NAME = "S2Coast_Russia_imported_points"

DATABASE_POINT_CRS = "EPSG:4326"

# Используется ТОЛЬКО при отсутствии CRS у файла точек.
# Не переопределяет CRS, которую прочитал GeoPandas.
POINTS_CRS_IF_MISSING: str | None = "EPSG:4326"

# Размер пакета вставки.
INSERT_BATCH_SIZE = 10_000

# True — удалить точные повторы lon/lat внутри импортируемого файла.
# Атрибуты и высота Z при сравнении не учитываются.
REMOVE_EXACT_DUPLICATES = True

# Защита от случайного повторного импорта.
ALLOW_DUPLICATE_DATASET_NAME = False

# Описание происхождения точек.
# Это только метаданные: никакого нового построения точек нет.
IMPORT_STRATEGY_NAME = "GeoJSONPointImport"


# ==================== ПУТИ ====================

def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()

    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved

    return resolved.resolve()


def resolve_input_file(
    path: str | Path,
    *,
    description: str,
) -> Path:
    resolved = resolve_project_path(path)

    if not resolved.is_file():
        raise FileNotFoundError(
            f"{description} не найден или не является файлом:\n"
            f"{resolved}"
        )

    return resolved


# ==================== ПРОВЕРКА БЕРЕГОВЫХ ЛИНИЙ ====================

def validate_coastline_file(
    path: Path,
    *,
    description: str,
) -> None:
    """
    Проверяет наличие непустых линейных геометрий.

    Файл не изменяется. Совпадение точек с линиями
    и расстояние от точек до линий здесь не проверяются.
    """
    logger.info("Проверка {}: {}", description, path)

    coastline = gpd.read_file(path)

    if coastline.empty:
        raise ValueError(f"{description}: файл не содержит объектов.")

    nonempty = (
        coastline.geometry.notna()
        & ~coastline.geometry.is_empty
    )

    if not nonempty.any():
        raise ValueError(
            f"{description}: нет непустых геометрий."
        )

    geometry_types = coastline.loc[
        nonempty, coastline.geometry.name
    ].geom_type

    supported = geometry_types.isin(
        ["LineString", "MultiLineString"]
    )

    if not supported.all():
        wrong_types = sorted(
            geometry_types.loc[~supported].unique().tolist()
        )

        raise TypeError(
            f"{description}: ожидались LineString/MultiLineString, "
            f"получены также {wrong_types}."
        )

    if coastline.crs is None:
        raise ValueError(
            f"{description}: не задана CRS. "
            "Задайте корректную систему координат "
            "в исходном файле до импорта."
        )

    logger.info(
        "{}: непустых объектов={}, CRS={}",
        description,
        int(nonempty.sum()),
        coastline.crs,
    )


# ==================== ЧТЕНИЕ ТОЧЕК ====================

def load_points_wgs84(path: Path):
    """
    Читает готовые точки и преобразует их в EPSG:4326.

    Возвращает:
        coordinates — массив N x 2 с колонками lon, lat;
        statistics — статистику импорта.
    """
    logger.info("Чтение точек: {}", path)

    points = gpd.read_file(path)

    input_count = len(points)

    if points.empty:
        raise ValueError("GeoJSON точек не содержит объектов.")

    if points.crs is None:
        if POINTS_CRS_IF_MISSING is None:
            raise ValueError(
                "У файла точек отсутствует CRS. "
                "Укажите POINTS_CRS_IF_MISSING, "
                "если исходная система координат известна."
            )

        points = points.set_crs(POINTS_CRS_IF_MISSING)

        logger.warning(
            "CRS отсутствовала: назначена {}",
            POINTS_CRS_IF_MISSING,
        )

    input_crs = str(points.crs)

    nonempty = (
        points.geometry.notna()
        & ~points.geometry.is_empty
    )

    empty_count = int((~nonempty).sum())

    if empty_count:
        logger.warning(
            "Пропущено пустых/отсутствующих геометрий: {}",
            empty_count,
        )

    points = points.loc[nonempty].copy()

    if points.empty:
        raise ValueError(
            "После удаления пустых геометрий точек не осталось."
        )

    wrong_type = points.geometry.geom_type != "Point"

    if wrong_type.any():
        wrong_types = sorted(
            points.loc[
                wrong_type, points.geometry.name
            ].geom_type.unique().tolist()
        )

        raise TypeError(
            "Файл должен содержать отдельные Point. "
            f"Найдены другие типы: {wrong_types}. "
            "MultiPoint предварительно разделите на отдельные точки."
        )

    points = points.to_crs(DATABASE_POINT_CRS)

    coordinates = np.column_stack(
        [
            points.geometry.x.to_numpy(dtype=float),
            points.geometry.y.to_numpy(dtype=float),
        ]
    )

    valid = (
        np.isfinite(coordinates).all(axis=1)
        & (np.abs(coordinates[:, 0]) <= 180.0)
        & (np.abs(coordinates[:, 1]) <= 90.0)
    )

    if not valid.all():
        invalid_positions = np.flatnonzero(~valid)

        raise ValueError(
            "После преобразования в EPSG:4326 найдены "
            f"некорректные координаты: {len(invalid_positions)}. "
            "Проверьте исходную CRS. "
            "Первые проблемные позиции среди непустых точек: "
            f"{invalid_positions[:10].tolist()}"
        )

    duplicate_count = 0

    if REMOVE_EXACT_DUPLICATES:
        # Сохраняем первое вхождение и исходный порядок.
        # +180 и -180 рассматриваем как один меридиан.
        seen = set()
        keep_indices = []

        for index, (lon, lat) in enumerate(coordinates):
            key_lon = -180.0 if lon == 180.0 else float(lon)
            key = (key_lon, float(lat))

            if key in seen:
                duplicate_count += 1
                continue

            seen.add(key)
            keep_indices.append(index)

        coordinates = coordinates[
            np.asarray(keep_indices, dtype=np.int64)
        ]

    if len(coordinates) == 0:
        raise ValueError("Нет точек для сохранения.")

    statistics = {
        "input_count": input_count,
        "empty_count": empty_count,
        "duplicates_removed": duplicate_count,
        "imported_count": len(coordinates),
        "input_crs": input_crs,
    }

    logger.info(
        "Точек в файле: {}; пустых: {}; повторов удалено: {}; "
        "к импорту: {}",
        input_count,
        empty_count,
        duplicate_count,
        len(coordinates),
    )

    return coordinates, statistics


# ==================== ИМПОРТ В SQLITE ====================

def main() -> None:
    if not DATASET_NAME.strip():
        raise ValueError("DATASET_NAME не должен быть пустым.")

    if (
        isinstance(INSERT_BATCH_SIZE, bool)
        or not isinstance(INSERT_BATCH_SIZE, int)
        or INSERT_BATCH_SIZE <= 0
    ):
        raise ValueError(
            "INSERT_BATCH_SIZE должен быть положительным целым."
        )

    points_path = resolve_input_file(
        POINTS_GEOJSON_PATH,
        description="GeoJSON точек",
    )

    main_path = resolve_input_file(
        COASTLINE_MAIN_PATH,
        description="Основная береговая линия",
    )

    other_path = (
        resolve_input_file(
            COASTLINE_OTHER_PATH,
            description="Дополнительная береговая линия",
        )
        if COASTLINE_OTHER_PATH is not None
        else None
    )

    database_path = resolve_project_path(DATABASE_PATH)

    input_paths = {points_path, main_path}

    if other_path is not None:
        input_paths.add(other_path)

    if database_path in input_paths:
        raise ValueError(
            "Путь к базе данных совпадает с одним из входных файлов."
        )

    # Сначала проверяем входные данные, затем создаём базу.
    validate_coastline_file(
        main_path,
        description="Основная береговая линия",
    )

    if other_path is not None:
        validate_coastline_file(
            other_path,
            description="Дополнительная береговая линия",
        )

    coordinates, statistics = load_points_wgs84(points_path)

    strategy_params = json.dumps(
        {
            "import_mode": "existing_geojson_points",
            "points_geojson_path": str(points_path),
            "input_crs": statistics["input_crs"],
            "database_point_crs": DATABASE_POINT_CRS,
            "remove_exact_duplicates": REMOVE_EXACT_DUPLICATES,
            "sequence_order": "input_feature_order",
            **statistics,
        },
        ensure_ascii=False,
    )

    database_path.parent.mkdir(parents=True, exist_ok=True)

    # Собственный engine гарантирует использование DATABASE_PATH,
    # даже если COASTLINE_DATABASE_URL в окружении указывает иначе.
    # Модуль storage.db здесь намеренно не используется.
    database_url = URL.create(
        drivername="sqlite",
        database=str(database_path),
    )

    engine = create_engine(
        database_url,
        echo=False,
        connect_args={"timeout": 60},
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()

        try:
            cursor.execute("PRAGMA foreign_keys = ON")
        finally:
            cursor.close()

    try:
        Base.metadata.create_all(engine)

        logger.info("База данных: {}", database_path)

        # Один источник и все его точки сохраняются
        # в рамках одной транзакции.
        with Session(engine) as session:
            with session.begin():
                existing_source_id = session.scalar(
                    select(CoastlineSourceModel.id)
                    .where(
                        CoastlineSourceModel.name == DATASET_NAME
                    )
                    .limit(1)
                )

                if (
                    existing_source_id is not None
                    and not ALLOW_DUPLICATE_DATASET_NAME
                ):
                    raise ValueError(
                        f"Источник {DATASET_NAME!r} уже существует: "
                        f"id={existing_source_id}. "
                        "Измените DATASET_NAME либо явно разрешите "
                        "повтор имени через "
                        "ALLOW_DUPLICATE_DATASET_NAME = True."
                    )

                source = CoastlineSourceModel(
                    name=DATASET_NAME,
                    strategy_name=IMPORT_STRATEGY_NAME,
                    source_mode="main_only",
                    strategy_params=strategy_params,
                    main_geojson_path=str(main_path),
                    other_geojson_path=(
                        str(other_path)
                        if other_path is not None
                        else None
                    ),
                    crs=DATABASE_POINT_CRS,
                    points_count=len(coordinates),
                )

                session.add(source)
                session.flush()

                source_id = int(source.id)
                total_count = len(coordinates)

                for start in range(
                    0,
                    total_count,
                    INSERT_BATCH_SIZE,
                ):
                    stop = min(
                        start + INSERT_BATCH_SIZE,
                        total_count,
                    )

                    batch = [
                        {
                            "source_id": source_id,
                            "seq": start + offset,
                            "lon": float(lon),
                            "lat": float(lat),
                        }
                        for offset, (lon, lat) in enumerate(
                            coordinates[start:stop]
                        )
                    ]

                    session.execute(
                        insert(CoastlinePointModel),
                        batch,
                    )

                    logger.info(
                        "Вставлено в транзакцию: {} / {}",
                        stop,
                        total_count,
                    )

        logger.success(
            "Импорт завершён: {} точек, source_id={}, name={!r}",
            len(coordinates),
            source_id,
            DATASET_NAME,
        )

        logger.info("GeoJSON точек: {}", points_path)
        logger.info("Береговая линия: {}", main_path)
        logger.info("CRS хранения: {}", DATABASE_POINT_CRS)
        logger.info("SQLite: {}", database_path)

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()