"""
db_pypeline/1_2_export_points_to_geojson.py

Экспорт готовых береговых точек из coastline.db в GeoJSON.

Соответствует текущему контракту импорта:
- coastline_points.lon / lat содержат координаты EPSG:4326;
- coastline_sources.crs содержит EPSG:4326;
- относительные пути разрешаются относительно корня проекта;
- DATABASE_PATH задаёт базу явно;
- COASTLINE_DATABASE_URL не используется.

Выбор источника:
- если SOURCE_ID задан — используется конкретный источник;
- иначе выбирается последний по id источник с SOURCE_NAME
  и дополнительными фильтрами STRATEGY_NAME / SOURCE_MODE.

Экспорт:
- порядок точек: seq, затем id;
- point_id — первичный ключ точки из SQLite;
- координаты и идентификаторы не изменяются;
- исходный файл базы не модифицируется;
- GeoJSON записывается постепенно, без загрузки всех точек в память.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

from loguru import logger
from pyproj import CRS


# ==================== КОРЕНЬ ПРОЕКТА ====================

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


# ==================== НАСТРОЙКИ ====================

# Тот же путь, что в предыдущем скрипте импорта.
DATABASE_PATH = Path(
    "big_scale/data/db/coastline.db"
)

OUTPUT_PATH = Path(
    "big_scale/data/coastline/exported_points.geojson"
)

SOURCE_NAME = "S2Coast_Russia_imported_points"

# None — не применять дополнительный фильтр.
STRATEGY_NAME: str | None = None
SOURCE_MODE: str | None = None

# Если задан, фильтры по имени, стратегии и режиму игнорируются.
SOURCE_ID: int | None = None

# Размер порции чтения из SQLite.
READ_BATCH_SIZE = 10_000

# Если False, существующий выходной файл не перезаписывается.
OVERWRITE_OUTPUT = True


# ==================== ПУТИ ====================

def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()

    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved

    return resolved.resolve()


# ==================== БАЗА ДАННЫХ ====================

def validate_schema(connection: sqlite3.Connection) -> None:
    required_columns = {
        "coastline_sources": {
            "id",
            "name",
            "strategy_name",
            "source_mode",
            "points_count",
            "crs",
            "created_at",
        },
        "coastline_points": {
            "id",
            "source_id",
            "seq",
            "lon",
            "lat",
        },
    }

    for table, expected in required_columns.items():
        # Имена таблиц заданы константами внутри скрипта.
        actual = {
            row["name"]
            for row in connection.execute(
                f'PRAGMA table_info("{table}")'
            )
        }

        missing = expected - actual

        if missing:
            raise ValueError(
                f"Таблица {table!r} отсутствует "
                "или имеет несовместимую схему. "
                f"Не найдены поля: {sorted(missing)}"
            )


def log_sources(connection: sqlite3.Connection) -> None:
    cursor = connection.execute(
        """
        SELECT
            id,
            name,
            strategy_name,
            source_mode,
            points_count,
            crs,
            created_at
        FROM coastline_sources
        ORDER BY id
        """
    )

    found = False

    logger.info("Источники в базе:")

    for source in cursor:
        found = True

        logger.info(
            "id={} | name={!r} | strategy={!r} | mode={!r} | "
            "points={} | crs={} | created={}",
            source["id"],
            source["name"],
            source["strategy_name"],
            source["source_mode"],
            source["points_count"],
            source["crs"],
            source["created_at"],
        )

    if not found:
        raise ValueError(
            "В базе нет источников. "
            "Сначала выполните импорт точек."
        )


def select_source(connection: sqlite3.Connection) -> sqlite3.Row:
    if SOURCE_ID is not None:
        source = connection.execute(
            """
            SELECT *
            FROM coastline_sources
            WHERE id = ?
            """,
            (SOURCE_ID,),
        ).fetchone()

        if source is None:
            raise ValueError(
                f"Источник с id={SOURCE_ID} не найден."
            )

        return source

    if not SOURCE_NAME or not SOURCE_NAME.strip():
        raise ValueError(
            "Задайте SOURCE_NAME либо конкретный SOURCE_ID."
        )

    conditions = ["name = ?"]
    parameters = [SOURCE_NAME]

    if STRATEGY_NAME is not None:
        conditions.append("strategy_name = ?")
        parameters.append(STRATEGY_NAME)

    if SOURCE_MODE is not None:
        conditions.append("source_mode = ?")
        parameters.append(SOURCE_MODE)

    query = f"""
        SELECT *
        FROM coastline_sources
        WHERE {" AND ".join(conditions)}
        ORDER BY id DESC
        LIMIT 1
    """

    source = connection.execute(
        query,
        parameters,
    ).fetchone()

    if source is None:
        raise ValueError(
            "Не найден источник: "
            f"name={SOURCE_NAME!r}, "
            f"strategy={STRATEGY_NAME!r}, "
            f"mode={SOURCE_MODE!r}."
        )

    return source


def validate_source_crs(source: sqlite3.Row) -> None:
    source_crs = source["crs"]

    if not source_crs:
        raise ValueError(
            f"У источника id={source['id']} не указана CRS."
        )

    try:
        parsed_crs = CRS.from_user_input(source_crs)
    except Exception as error:
        raise ValueError(
            f"Некорректная CRS источника: {source_crs!r}"
        ) from error

    expected_crs = CRS.from_epsg(4326)

    if not parsed_crs.equals(
        expected_crs,
        ignore_axis_order=True,
    ):
        raise ValueError(
            f"CRS источника — {source_crs!r}, "
            "но текущий контракт хранения требует EPSG:4326. "
            "Экспорт остановлен, чтобы не записать "
            "неверные координаты в GeoJSON."
        )


# ==================== ЭКСПОРТ ====================

def export_points(
    connection: sqlite3.Connection,
    source: sqlite3.Row,
    output_path: Path,
) -> int:
    source_id = int(source["id"])

    actual_count = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM coastline_points
            WHERE source_id = ?
            """,
            (source_id,),
        ).fetchone()[0]
    )

    if actual_count == 0:
        raise ValueError(
            f"Источник id={source_id} не содержит точек."
        )

    metadata_count = int(source["points_count"])

    if actual_count != metadata_count:
        logger.warning(
            "Количество точек в метаданных ({}) "
            "не совпадает с фактическим ({}). "
            "Будут экспортированы фактические записи.",
            metadata_count,
            actual_count,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    cursor = connection.execute(
        """
        SELECT id, source_id, seq, lon, lat
        FROM coastline_points
        WHERE source_id = ?
        ORDER BY seq, id
        """,
        (source_id,),
    )

    written_count = 0

    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            stream.write(
                '{"type":"FeatureCollection","features":[\n'
            )

            while True:
                batch = cursor.fetchmany(READ_BATCH_SIZE)

                if not batch:
                    break

                for row in batch:
                    point_id = int(row["id"])

                    if row["lon"] is None or row["lat"] is None:
                        raise ValueError(
                            f"Точка id={point_id}: "
                            "отсутствуют lon или lat."
                        )

                    lon = float(row["lon"])
                    lat = float(row["lat"])

                    if not (
                        math.isfinite(lon)
                        and math.isfinite(lat)
                        and -180.0 <= lon <= 180.0
                        and -90.0 <= lat <= 90.0
                    ):
                        raise ValueError(
                            f"Точка id={point_id}: "
                            f"некорректные координаты "
                            f"lon={lon}, lat={lat}."
                        )

                    feature = {
                        "type": "Feature",
                        "id": point_id,
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat],
                        },
                        "properties": {
                            # id сохраняется также для совместимости
                            # с форматом прежнего экспортера.
                            "id": point_id,
                            "point_id": point_id,
                            "source_id": int(row["source_id"]),
                            "seq": int(row["seq"]),
                        },
                    }

                    if written_count:
                        stream.write(",\n")

                    json.dump(
                        feature,
                        stream,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )

                    written_count += 1

                logger.info(
                    "Экспортировано: {} / {}",
                    written_count,
                    actual_count,
                )

            stream.write("\n]}\n")

        if written_count != actual_count:
            raise RuntimeError(
                f"Ожидалось {actual_count} точек, "
                f"записано {written_count}."
            )

        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    finally:
        cursor.close()

    return written_count


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main() -> None:
    if (
        isinstance(READ_BATCH_SIZE, bool)
        or not isinstance(READ_BATCH_SIZE, int)
        or READ_BATCH_SIZE <= 0
    ):
        raise ValueError(
            "READ_BATCH_SIZE должен быть положительным целым."
        )

    database_path = resolve_project_path(DATABASE_PATH)
    output_path = resolve_project_path(OUTPUT_PATH)

    if not database_path.is_file():
        raise FileNotFoundError(
            f"База данных не найдена:\n{database_path}"
        )

    if output_path == database_path:
        raise ValueError(
            "Выходной файл не должен совпадать с базой данных."
        )

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    if temporary_path == database_path:
        raise ValueError(
            "Путь временного файла совпадает с базой данных."
        )

    if output_path.exists() and not OVERWRITE_OUTPUT:
        raise FileExistsError(
            f"Выходной файл уже существует:\n{output_path}"
        )

    logger.info("База данных: {}", database_path)
    logger.info("Выходной GeoJSON: {}", output_path)

    uri = database_path.as_uri() + "?mode=ro"

    with closing(
        sqlite3.connect(
            uri,
            uri=True,
            timeout=60,
        )
    ) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")

        # Единый снимок данных для выбора источника,
        # подсчёта и экспорта точек.
        connection.execute("BEGIN")

        try:
            validate_schema(connection)
            log_sources(connection)

            source = select_source(connection)
            validate_source_crs(source)

            logger.info(
                "Выбран источник: id={}, name={!r}, "
                "strategy={!r}, mode={!r}, crs={}",
                source["id"],
                source["name"],
                source["strategy_name"],
                source["source_mode"],
                source["crs"],
            )

            written_count = export_points(
                connection=connection,
                source=source,
                output_path=output_path,
            )

        finally:
            # Завершаем читающую транзакцию, не изменяя базу.
            connection.rollback()

    logger.success(
        "Готово. Экспортировано {} точек: {}",
        written_count,
        output_path,
    )


if __name__ == "__main__":
    main()