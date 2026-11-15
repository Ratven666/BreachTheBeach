"""
big_scale/pypelin/2_2_export_normals_to_geojson.py

Экспорт сохранённых нормалей из coastline.db в GeoJSON.

Использует существующий SQLiteNormalsToGeoJsonExporter.
Базовые файлы src/ не изменяются.

Контракт:
- точки coastline_points.lon / lat хранятся в EPSG:4326;
- nx/ny относятся к working_crs набора нормалей;
- конец нормали строится в working_crs;
- итоговая геометрия записывается в EPSG:4326;
- относительные пути разрешаются от корня проекта;
- DATABASE_PATH имеет приоритет над переменной окружения;
- экспортируются только существующие записи coastline_normals.

Выбор набора:
- SOURCE_ID задан: используется конкретный набор;
- SOURCE_ID = None: последний по id набор с SOURCE_NAME;
- SEA_SIDE — дополнительный фильтр набора, не отдельных нормалей.

Длина NORMAL_LENGTH_M задаётся в метрах рабочей проекции.
Она не обязана точно совпадать с геодезической длиной
экспортированного отрезка на эллипсоиде.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from loguru import logger
from pyproj import CRS
from sqlalchemy import func, select
from sqlalchemy.engine import URL, make_url


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


# ==================== НАСТРОЙКИ ====================

DATABASE_PATH = Path(
    "big_scale/data/db/coastline.db"
)

OUTPUT_PATH = Path(
    "big_scale/data/coastline/exported_normals.geojson"
)

SOURCE_NAME = "S2Coast_normals_land_checked"

# None — не фильтровать по стороне.
# Возможные значения: None, "left", "right", "mixed".
#
# Это фильтр поля coastline_normal_sources.sea_side,
# а не отбор отдельных векторов внутри mixed-набора.
SEA_SIDE: str | None = None

# Если задан — SOURCE_NAME и SEA_SIDE игнорируются.
SOURCE_ID: int | None = None

NORMAL_LENGTH_M = 500.0

OVERWRITE_OUTPUT = True


# ==================== ПУТИ ====================

def resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()

    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved

    return resolved.resolve()


# ==================== ПРОВЕРКА НАСТРОЕК ====================

def validate_settings() -> None:
    if (
        isinstance(NORMAL_LENGTH_M, bool)
        or not isinstance(NORMAL_LENGTH_M, (int, float))
        or not math.isfinite(NORMAL_LENGTH_M)
        or NORMAL_LENGTH_M <= 0
    ):
        raise ValueError(
            "NORMAL_LENGTH_M должен быть положительным "
            "конечным числом."
        )

    if SOURCE_ID is not None:
        if (
            isinstance(SOURCE_ID, bool)
            or not isinstance(SOURCE_ID, int)
            or SOURCE_ID <= 0
        ):
            raise ValueError(
                "SOURCE_ID должен быть положительным целым "
                "или None."
            )

    else:
        if not SOURCE_NAME or not SOURCE_NAME.strip():
            raise ValueError(
                "Задайте SOURCE_NAME либо конкретный SOURCE_ID."
            )

        if SEA_SIDE not in {None, "left", "right", "mixed"}:
            raise ValueError(
                "SEA_SIDE должен быть None, left, right или mixed."
            )


# ==================== ПОДКЛЮЧЕНИЕ К БАЗЕ ====================

def setup_database_url(database_path: Path) -> None:
    """
    Выставляет URL до импорта storage.db и экспортера.

    Уже существующая переменная окружения не переопределяет
    явно заданный DATABASE_PATH.
    """
    if not database_path.is_file():
        raise FileNotFoundError(
            f"База данных не найдена:\n{database_path}"
        )

    url = URL.create(
        drivername="sqlite",
        database=str(database_path),
    ).render_as_string(hide_password=False)

    previous = os.environ.get(
        "COASTLINE_DATABASE_URL"
    )

    if previous is not None and previous != url:
        logger.warning(
            "COASTLINE_DATABASE_URL переопределён: "
            "используется DATABASE_PATH из этого скрипта."
        )

    os.environ["COASTLINE_DATABASE_URL"] = url

    logger.debug(
        "COASTLINE_DATABASE_URL -> {}",
        url,
    )


# ==================== ОСНОВНОЙ ЭКСПОРТ ====================

def main() -> None:
    validate_settings()

    database_path = resolve_path(DATABASE_PATH)
    output_path = resolve_path(OUTPUT_PATH)

    if output_path == database_path:
        raise ValueError(
            "Выходной файл совпадает с базой данных."
        )

    if output_path.exists() and not OVERWRITE_OUTPUT:
        raise FileExistsError(
            f"Выходной файл уже существует:\n{output_path}"
        )

    # Пишем сначала в отдельный GeoJSON.
    # Итоговый файл заменяем после успешного экспорта.
    temporary_path = output_path.with_name(
        output_path.stem + ".tmp.geojson"
    )

    if temporary_path in {database_path, output_path}:
        raise ValueError(
            "Некорректный путь временного файла."
        )

    setup_database_url(database_path)

    # Важно: эти импорты выполняются после настройки URL.
    from src.coastline.storage import db as _db

    from src.coastline.storage.models import (
        CoastlineNormalModel,
        CoastlinePointModel,
        CoastlineSourceModel,
    )

    from src.coastline.exporters.SQLiteNormalsToGeoJsonExporter import (
        SQLiteNormalsToGeoJsonExporter,
    )

    # Если storage.db уже был импортирован другим модулем,
    # изменение env не переключает существующий engine.
    actual_database = make_url(
        str(_db.engine.url)
    ).database

    if (
        actual_database is None
        or resolve_path(actual_database) != database_path
    ):
        raise RuntimeError(
            "Модуль storage.db уже подключён к другой базе: "
            f"{_db.engine.url}. "
            "Запустите этот скрипт отдельным процессом."
        )

    logger.info(
        "База данных: {}",
        database_path,
    )

    exporter = SQLiteNormalsToGeoJsonExporter()

    sources = exporter.list_sources()

    if not sources:
        raise ValueError(
            "В базе нет наборов нормалей. "
            "Сначала выполните расчёт нормалей."
        )

    logger.info(
        "Наборов нормалей в БД: {}",
        len(sources),
    )

    for source in sources:
        logger.info(
            "id={} | name={!r} | point_source_id={} | "
            "sea_side={!r} | normals={} | "
            "working_crs={!r} | created={}",
            source["id"],
            source["name"],
            source["point_source_id"],
            source["sea_side"],
            source["normals_count"],
            source["working_crs"],
            source["created_at"],
        )

    if SOURCE_ID is not None:
        candidates = [
            source
            for source in sources
            if int(source["id"]) == SOURCE_ID
        ]

    else:
        candidates = [
            source
            for source in sources
            if source["name"] == SOURCE_NAME
            and (
                SEA_SIDE is None
                or source["sea_side"] == SEA_SIDE
            )
        ]

    if not candidates:
        raise ValueError(
            "Подходящий набор нормалей не найден: "
            f"SOURCE_ID={SOURCE_ID!r}, "
            f"SOURCE_NAME={SOURCE_NAME!r}, "
            f"SEA_SIDE={SEA_SIDE!r}."
        )

    selected = max(
        candidates,
        key=lambda item: int(item["id"]),
    )

    normal_source_id = int(selected["id"])
    point_source_id = int(selected["point_source_id"])

    working_crs_value = selected["working_crs"]

    if not working_crs_value:
        raise ValueError(
            f"У набора id={normal_source_id} "
            "отсутствует working_crs. "
            "Нельзя корректно интерпретировать nx/ny."
        )

    working_crs = CRS.from_user_input(
        working_crs_value
    )

    if (
        not working_crs.is_projected
        or len(working_crs.axis_info) < 2
        or any(
            not math.isclose(
                axis.unit_conversion_factor,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for axis in working_crs.axis_info[:2]
        )
    ):
        raise ValueError(
            "working_crs должна быть проекционной CRS "
            "с единицами в метрах."
        )

    with _db.SessionLocal() as session:
        point_source = session.get(
            CoastlineSourceModel,
            point_source_id,
        )

        if point_source is None:
            raise ValueError(
                f"Источник точек id={point_source_id} не найден."
            )

        if (
            point_source.crs is None
            or not CRS.from_user_input(point_source.crs).equals(
                CRS.from_epsg(4326),
                ignore_axis_order=True,
            )
        ):
            raise ValueError(
                "CRS источника точек должна соответствовать "
                "EPSG:4326."
            )

        actual_normals_count = int(
            session.scalar(
                select(func.count())
                .select_from(CoastlineNormalModel)
                .where(
                    CoastlineNormalModel.normal_source_id
                    == normal_source_id
                )
            )
        )

        joined_count = int(
            session.scalar(
                select(func.count())
                .select_from(CoastlineNormalModel)
                .join(
                    CoastlinePointModel,
                    CoastlineNormalModel.point_id
                    == CoastlinePointModel.id,
                )
                .where(
                    CoastlineNormalModel.normal_source_id
                    == normal_source_id,
                    CoastlinePointModel.source_id
                    == point_source_id,
                )
            )
        )

        if actual_normals_count == 0:
            raise ValueError(
                f"Набор id={normal_source_id} не содержит нормалей."
            )

        if joined_count != actual_normals_count:
            raise ValueError(
                "Нарушены связи нормалей с точками: "
                f"нормалей={actual_normals_count}, "
                "точек правильного источника="
                f"{joined_count}. Экспорт остановлен."
            )

        if (
            actual_normals_count
            != int(selected["normals_count"])
        ):
            raise ValueError(
                "Количество нормалей не совпадает "
                "с метаданными: "
                f"metadata={selected['normals_count']}, "
                f"actual={actual_normals_count}."
            )

    logger.info(
        "Выбран набор: id={}, name={!r}, "
        "sea_side={!r}, normals={}",
        normal_source_id,
        selected["name"],
        selected["sea_side"],
        actual_normals_count,
    )

    logger.info(
        "Рабочая CRS: {}; длина в рабочей проекции: {} м",
        working_crs_value,
        NORMAL_LENGTH_M,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        logger.info(
            "Построение и экспорт линий нормалей..."
        )

        exporter.export_by_id(
            normal_source_id=normal_source_id,
            output_path=temporary_path,
            normal_length_m=NORMAL_LENGTH_M,
        )

        if (
            not temporary_path.is_file()
            or temporary_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Экспортер не создал непустой GeoJSON."
            )

        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    logger.success(
        "Экспортировано {} нормалей: {}",
        actual_normals_count,
        output_path,
    )


if __name__ == "__main__":
    main()