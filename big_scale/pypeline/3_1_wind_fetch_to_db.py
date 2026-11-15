"""
big_scale/pypelin/3_1_wind_fetch_to_db.py

Расчёт wind fetch по сохранённым нормалям и запись в SQLite.

Особенности:
- относительные пути разрешаются от корня проекта;
- база задаётся явно через DATABASE_PATH;
- загружаются только точки выбранного набора нормалей;
- используется working_crs этого набора;
- базовые файлы src/ не изменяются;
- расчёт выполняется порциями точек;
- промежуточные результаты хранятся во временной SQLite;
- основная wind_fetches обновляется одной транзакцией.

В таблицу wind_fetches попадают только направления,
не исключённые калькулятором как сушевой сектор.

Ограничение текущего калькулятора:
лучи и длины рассчитываются в рабочей плоской проекции.
azimuth_deg — направление относительно её координатного севера,
а не автоматически истинный геодезический азимут.

В текущей схеме wind_fetches нет normal_source_id.
Новые результаты заменяют предыдущие fetch для тех же point_id.
"""

from __future__ import annotations

import math
import sqlite3
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from pyproj import CRS
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
        "Не найден корень BreachTheBeach."
    )


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.coastline.storage.models import (  # noqa: E402
    Base,
    CoastlineNormalModel,
    CoastlineNormalSourceModel,
    CoastlinePointModel,
    CoastlineSourceModel,
    WindFetchModel,
)
from src.wind_fetch.SequentialMultiDirectionFetchCalculator import (  # noqa: E402
    SequentialMultiDirectionFetchCalculator,
)
from src.wind_fetch.WindFetchConfig import WindFetchConfig  # noqa: E402
from src.wind_fetch.models import WindFetchPaths  # noqa: E402


# ==================== НАСТРОЙКИ ====================

DATABASE_PATH = Path(
    "big_scale/data/db/coastline.db"
)

# Если задан — используется конкретный набор.
NORMAL_SOURCE_ID: int | None = None

# Если NORMAL_SOURCE_ID = None, выбирается последний
# набор с таким именем.
# None — последний набор независимо от имени.
NORMAL_SOURCE_NAME: str | None = (
    "S2Coast_normals_land_checked"
)

AZIMUTH_STEP_DEG = 1.0

DEFAULT_FETCH_M = 100_000.0
DEFAULT_OFFSET_M = 1.0

SHOW_PROGRESS = True

# Размер порции расчёта.
# При 100 точках и шаге 1° одновременно создаётся
# не более примерно 36 000 результатов.
POINTS_PER_BATCH = 100

INSERT_BATCH_SIZE = 10_000
SQL_ID_BATCH_SIZE = 500

# Расширенная линия вместо основной из метаданных.
# Оригинальная и дополнительная линии при этом
# также сохраняются в геометрии препятствий.
EXTENDED_COASTLINE_PATH: Path | None = Path(
    "big_scale/data/S2CoastRussiaCropped.geojson"
)

# Пример:
# EXTENDED_COASTLINE_PATH = Path(
#     "big_scale/data/coastline_extended.geojson"
# )

# Рабочая директория временного расчёта.
# None — системный временный каталог.
TEMP_PARENT_DIR: Path | None = None


# ==================== ПУТИ ====================

def resolve_path(path: str | Path) -> Path:
    result = Path(path).expanduser()

    if not result.is_absolute():
        result = PROJECT_ROOT / result

    return result.resolve()


def require_file(path, description):
    result = resolve_path(path)

    if not result.is_file():
        raise FileNotFoundError(
            f"{description} не найден:\n{result}"
        )

    return result


# ==================== ПРОВЕРКИ ====================

def validate_settings():
    if (
        not math.isfinite(AZIMUTH_STEP_DEG)
        or not 0 < AZIMUTH_STEP_DEG <= 90
    ):
        raise ValueError(
            "AZIMUTH_STEP_DEG должен быть > 0 и <= 90."
        )

    direction_count = round(
        360.0 / AZIMUTH_STEP_DEG
    )

    if not math.isclose(
        direction_count * AZIMUTH_STEP_DEG,
        360.0,
        rel_tol=0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "AZIMUTH_STEP_DEG должен делить 360 без остатка."
        )

    for name, value in (
        ("DEFAULT_FETCH_M", DEFAULT_FETCH_M),
        ("DEFAULT_OFFSET_M", DEFAULT_OFFSET_M),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{name} должен быть положительным."
            )

    for name, value in (
        ("POINTS_PER_BATCH", POINTS_PER_BATCH),
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


def angular_distance(a, b):
    return abs(
        (float(a) - float(b) + 180.0) % 360.0 - 180.0
    )


# ==================== АДАПТЕР КАЛЬКУЛЯТОРА ====================

class PipelineFetchCalculator(
    SequentialMultiDirectionFetchCalculator
):
    """
    Принудительно использует CRS сохранённых нормалей.

    Файлы базового калькулятора не изменяются.
    """

    def __init__(self, *, working_crs, **kwargs):
        self._pipeline_working_crs = CRS.from_user_input(
            working_crs
        )

        super().__init__(**kwargs)

    def _resolve_metric_crs(self):
        return self._pipeline_working_crs


# ==================== ЗАГРУЗКА ИЗ SQLITE ====================

def load_normals(engine):
    with Session(engine) as session:
        if NORMAL_SOURCE_ID is not None:
            source = session.get(
                CoastlineNormalSourceModel,
                NORMAL_SOURCE_ID,
            )

        else:
            query = select(CoastlineNormalSourceModel)

            if NORMAL_SOURCE_NAME is not None:
                query = query.where(
                    CoastlineNormalSourceModel.name
                    == NORMAL_SOURCE_NAME
                )

            source = session.scalars(
                query.order_by(
                    CoastlineNormalSourceModel.id.desc()
                ).limit(1)
            ).first()

        if source is None:
            raise ValueError(
                "Набор нормалей не найден. "
                "Сначала выполните расчёт нормалей."
            )

        point_source = session.get(
            CoastlineSourceModel,
            source.point_source_id,
        )

        if point_source is None:
            raise ValueError(
                "Источник береговых точек не найден."
            )

        if (
            point_source.crs is None
            or not CRS.from_user_input(point_source.crs).equals(
                CRS.from_epsg(4326),
                ignore_axis_order=True,
            )
        ):
            raise ValueError(
                "Точки SQLite должны храниться в EPSG:4326."
            )

        if not source.working_crs:
            raise ValueError(
                "У набора нормалей отсутствует working_crs."
            )

        working_crs = CRS.from_user_input(
            source.working_crs
        )

        if (
            not working_crs.is_projected
            or len(working_crs.axis_info) < 2
            or any(
                not math.isclose(
                    axis.unit_conversion_factor,
                    1.0,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                for axis in working_crs.axis_info[:2]
            )
        ):
            raise ValueError(
                "working_crs должна быть проекционной "
                "с единицами в метрах."
            )

        rows = session.execute(
            select(
                CoastlineNormalModel.id.label("normal_id"),
                CoastlineNormalModel.point_id,
                CoastlineNormalModel.nx,
                CoastlineNormalModel.ny,
                CoastlineNormalModel.normal_azimuth_deg,
                CoastlinePointModel.source_id.label(
                    "point_source_id"
                ),
                CoastlinePointModel.seq,
                CoastlinePointModel.lon,
                CoastlinePointModel.lat,
            )
            .join(
                CoastlinePointModel,
                CoastlineNormalModel.point_id
                == CoastlinePointModel.id,
            )
            .where(
                CoastlineNormalModel.normal_source_id
                == source.id
            )
            .order_by(
                CoastlinePointModel.seq,
                CoastlinePointModel.id,
            )
        ).all()

        metadata = {
            "id": int(source.id),
            "name": str(source.name),
            "point_source_id": int(source.point_source_id),
            "working_crs": working_crs,
            "expected_normals_count": int(source.normals_count),
            "main_path": point_source.main_geojson_path,
            "other_path": point_source.other_geojson_path,
        }

    if not rows:
        raise ValueError("Набор нормалей пуст.")

    if len(rows) != metadata["expected_normals_count"]:
        raise ValueError(
            "Количество доступных нормалей не соответствует "
            "метаданным: "
            f"metadata={metadata['expected_normals_count']}, "
            f"rows={len(rows)}."
        )

    records = []
    lons = []
    lats = []
    seen = set()

    for row in rows:
        point_id = int(row.point_id)

        if point_id in seen:
            raise ValueError(
                f"Повторяющаяся нормаль point_id={point_id}."
            )

        seen.add(point_id)

        if int(row.point_source_id) != metadata["point_source_id"]:
            raise ValueError(
                f"Точка {point_id} принадлежит другому источнику."
            )

        lon = float(row.lon)
        lat = float(row.lat)
        nx = float(row.nx)
        ny = float(row.ny)
        azimuth = float(row.normal_azimuth_deg)

        if not all(
            math.isfinite(value)
            for value in (lon, lat, nx, ny, azimuth)
        ):
            raise ValueError(
                f"Неконечные значения у point_id={point_id}."
            )

        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(
                f"Некорректные lon/lat у point_id={point_id}."
            )

        if not math.isclose(
            math.hypot(nx, ny),
            1.0,
            abs_tol=1e-6,
            rel_tol=0,
        ):
            raise ValueError(
                f"Неединичная нормаль point_id={point_id}."
            )

        expected_azimuth = (
            math.degrees(math.atan2(nx, ny)) + 360.0
        ) % 360.0

        if angular_distance(
            azimuth,
            expected_azimuth,
        ) > 1e-5:
            raise ValueError(
                f"Азимут не соответствует nx/ny: "
                f"point_id={point_id}."
            )

        records.append(
            {
                "point_id": point_id,
                "seq": int(row.seq),
                "normal_azimuth_deg": azimuth % 360.0,
            }
        )

        lons.append(lon)
        lats.append(lat)

    gdf = gpd.GeoDataFrame(
        records,
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326",
    )

    return metadata, gdf


# ==================== БЕРЕГОВЫЕ ЛИНИИ ====================

def prepare_coastline_paths(metadata, temporary_dir):
    if not metadata["main_path"]:
        raise ValueError(
            "У источника точек отсутствует main_geojson_path."
        )

    original_main = require_file(
        metadata["main_path"],
        "Основная береговая линия",
    )

    original_other = (
        require_file(
            metadata["other_path"],
            "Дополнительная береговая линия",
        )
        if metadata["other_path"]
        else None
    )

    if EXTENDED_COASTLINE_PATH is None:
        return (
            original_main,
            original_other,
        )

    extended = require_file(
        EXTENDED_COASTLINE_PATH,
        "Расширенная береговая линия",
    )

    # Не теряем дополнительную линию из метаданных:
    # все оригинальные препятствия передаются как other.
    auxiliary_paths = []

    for path in (original_main, original_other):
        if (
            path is not None
            and path != extended
            and path not in auxiliary_paths
        ):
            auxiliary_paths.append(path)

    if not auxiliary_paths:
        return extended, None

    if len(auxiliary_paths) == 1:
        return extended, auxiliary_paths[0]

    frames = []

    for path in auxiliary_paths:
        frame = gpd.read_file(path)

        if frame.crs is None:
            raise ValueError(
                f"У береговой линии отсутствует CRS: {path}"
            )

        frame = frame.to_crs("EPSG:4326")

        frames.append(
            frame[[frame.geometry.name]]
            .rename_geometry("geometry")
            if frame.geometry.name != "geometry"
            else frame[["geometry"]]
        )

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )

    combined_path = (
        temporary_dir / "original_coastlines.geojson"
    )

    combined.to_file(
        combined_path,
        driver="GeoJSON",
        index=False,
    )

    return extended, combined_path


# ==================== ПРОМЕЖУТОЧНЫЙ РАСЧЁТ ====================

def calculate_to_staging(
    calculator,
    normals_gdf,
    staging_connection,
):
    staging_connection.execute(
        """
        CREATE TABLE fetch_stage (
            point_id INTEGER NOT NULL,
            azimuth_deg REAL NOT NULL,
            fetch_length_m INTEGER NOT NULL,
            PRIMARY KEY (point_id, azimuth_deg)
        ) WITHOUT ROWID
        """
    )

    full_metric_points = (
        calculator.points_gdf_metric.copy()
    )

    expected_ids = normals_gdf["point_id"].astype(int).tolist()

    if (
        len(full_metric_points) != len(expected_ids)
        or full_metric_points["point_id"].astype(int).tolist()
        != expected_ids
    ):
        raise RuntimeError(
            "Калькулятор изменил состав или порядок точек."
        )

    expected_id_set = set(expected_ids)
    calculated_id_set = set()

    total_rows = 0

    with tqdm(
        total=len(full_metric_points),
        desc="Расчёт fetch по точкам",
        unit="точ.",
        dynamic_ncols=True,
        disable=not SHOW_PROGRESS,
    ) as progress:
        try:
            for start in range(
                0,
                len(full_metric_points),
                POINTS_PER_BATCH,
            ):
                batch_points = full_metric_points.iloc[
                    start:start + POINTS_PER_BATCH
                ].copy()

                calculator.points_gdf_metric = batch_points

                # Внутренний индикатор показывает лучи текущей порции.
                results = calculator.calculate(
                    show_progress=SHOW_PROGRESS,
                    log_every_points=0,
                )

                batch_ids = set(
                    batch_points["point_id"].astype(int)
                )

                produced_ids = set()
                insert_rows = []

                for result in results:
                    if result.skipped_by_land_sector:
                        continue

                    if angular_distance(
                        result.azimuth_deg,
                        result.normal_azimuth_deg,
                    ) > 90.0 + 1e-9:
                        continue

                    point_id = int(result.point_id)
                    azimuth = float(result.azimuth_deg) % 360.0
                    length = float(result.fetch_length_m)

                    if point_id not in batch_ids:
                        raise RuntimeError(
                            "Калькулятор вернул чужой point_id: "
                            f"{point_id}"
                        )

                    if (
                        not math.isfinite(azimuth)
                        or not math.isfinite(length)
                        or length < 0
                    ):
                        raise ValueError(
                            f"Некорректный fetch: point_id={point_id}."
                        )

                    insert_rows.append(
                        (
                            point_id,
                            azimuth,
                            int(round(length)),
                        )
                    )

                    produced_ids.add(point_id)

                if produced_ids != batch_ids:
                    missing = sorted(
                        batch_ids - produced_ids
                    )

                    raise RuntimeError(
                        "Не получены морские направления "
                        f"для точек: {missing[:20]}"
                    )

                staging_connection.executemany(
                    """
                    INSERT INTO fetch_stage (
                        point_id,
                        azimuth_deg,
                        fetch_length_m
                    )
                    VALUES (?, ?, ?)
                    """,
                    insert_rows,
                )

                staging_connection.commit()

                total_rows += len(insert_rows)
                calculated_id_set.update(produced_ids)

                progress.update(len(batch_points))
                progress.set_postfix(
                    rows=total_rows,
                    refresh=False,
                )

                del results
                del insert_rows

        finally:
            calculator.points_gdf_metric = full_metric_points

    if calculated_id_set != expected_id_set:
        raise RuntimeError(
            "Набор рассчитанных точек не соответствует исходному."
        )

    return total_rows


# ==================== ЗАПИСЬ В ОСНОВНУЮ БАЗУ ====================

def replace_fetches(
    engine,
    staging_connection,
    metadata,
    point_ids,
    total_rows,
):
    with Session(engine) as session:
        with session.begin():
            # В текущем пайплайне SQLite используется
            # с обычным режимом sqlite3 в Python 3.12.
            session.connection().exec_driver_sql(
                "BEGIN IMMEDIATE"
            )

            source = session.get(
                CoastlineNormalSourceModel,
                metadata["id"],
            )

            if source is None:
                raise RuntimeError(
                    "Набор нормалей удалён во время расчёта."
                )

            current_ids = set(
                session.scalars(
                    select(CoastlineNormalModel.point_id)
                    .join(
                        CoastlinePointModel,
                        CoastlineNormalModel.point_id
                        == CoastlinePointModel.id,
                    )
                    .where(
                        CoastlineNormalModel.normal_source_id
                        == metadata["id"],
                        CoastlinePointModel.source_id
                        == metadata["point_source_id"],
                    )
                ).all()
            )

            if current_ids != set(point_ids):
                raise RuntimeError(
                    "Состав точек/нормалей изменился "
                    "во время расчёта. Запись отменена."
                )

            with tqdm(
                total=len(point_ids),
                desc="Удаление прежних fetch",
                unit="точ.",
                dynamic_ncols=True,
                disable=not SHOW_PROGRESS,
            ) as progress:
                for start in range(
                    0,
                    len(point_ids),
                    SQL_ID_BATCH_SIZE,
                ):
                    batch = point_ids[
                        start:start + SQL_ID_BATCH_SIZE
                    ]

                    session.execute(
                        delete(WindFetchModel).where(
                            WindFetchModel.point_id.in_(batch)
                        ),
                        execution_options={
                            "synchronize_session": False
                        },
                    )

                    progress.update(len(batch))

            cursor = staging_connection.execute(
                """
                SELECT point_id, azimuth_deg, fetch_length_m
                FROM fetch_stage
                ORDER BY point_id, azimuth_deg
                """
            )

            written = 0

            with tqdm(
                total=total_rows,
                desc="Запись wind_fetches",
                unit="стр.",
                dynamic_ncols=True,
                disable=not SHOW_PROGRESS,
            ) as progress:
                while True:
                    rows = cursor.fetchmany(
                        INSERT_BATCH_SIZE
                    )

                    if not rows:
                        break

                    batch = [
                        {
                            "point_id": int(row[0]),
                            "azimuth_deg": float(row[1]),
                            "fetch_length_m": int(row[2]),
                        }
                        for row in rows
                    ]

                    session.execute(
                        insert(WindFetchModel),
                        batch,
                    )

                    written += len(batch)
                    progress.update(len(batch))

            if written != total_rows:
                raise RuntimeError(
                    "Количество записанных строк "
                    "не совпадает с рассчитанным."
                )

            logger.info(
                "Фиксация транзакции SQLite..."
            )


# ==================== ОСНОВНОЙ ПАЙПЛАЙН ====================

def main():
    validate_settings()

    database_path = require_file(
        DATABASE_PATH,
        "База данных",
    )

    temporary_parent = (
        resolve_path(TEMP_PARENT_DIR)
        if TEMP_PARENT_DIR is not None
        else None
    )

    if temporary_parent is not None:
        temporary_parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    engine = create_engine(
        URL.create(
            "sqlite",
            database=str(database_path),
        ),
        connect_args={"timeout": 60},
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(connection, connection_record):
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
        finally:
            cursor.close()

    try:
        Base.metadata.create_all(engine)

        metadata, normals_gdf = load_normals(
            engine
        )

        direction_count = round(
            360.0 / AZIMUTH_STEP_DEG
        )

        all_azimuths = [
            round(i * AZIMUTH_STEP_DEG, 10)
            for i in range(direction_count)
        ]

        if len(set(all_azimuths)) != len(all_azimuths):
            raise ValueError(
                "После округления появились повторяющиеся азимуты."
            )

        logger.info(
            "База: {}",
            database_path,
        )

        logger.info(
            "Нормали: id={}, name={!r}, points={}, CRS={}",
            metadata["id"],
            metadata["name"],
            len(normals_gdf),
            metadata["working_crs"],
        )

        logger.warning(
            "Используется плоский калькулятор: "
            "направления и длины относятся к working_crs. "
            "Это не геодезическая трассировка."
        )

        with tempfile.TemporaryDirectory(
            prefix="breachthebeach_fetch_",
            dir=(
                str(temporary_parent)
                if temporary_parent is not None
                else None
            ),
        ) as temporary:
            temporary_dir = Path(temporary)

            main_path, other_path = (
                prepare_coastline_paths(
                    metadata,
                    temporary_dir,
                )
            )

            logger.info(
                "Основная линия: {}",
                main_path,
            )

            logger.info(
                "Дополнительная линия: {}",
                other_path,
            )

            normals_path = (
                temporary_dir / "normals.geojson"
            )

            normals_gdf.to_file(
                normals_path,
                driver="GeoJSON",
                index=False,
            )

            paths = WindFetchPaths(
                main_coastline_path=str(main_path),
                other_coastline_path=(
                    str(other_path)
                    if other_path is not None
                    else None
                ),
                points_with_normals_path=str(normals_path),
            )

            config = WindFetchConfig(
                default_fetch_m=DEFAULT_FETCH_M,
                default_offset_m=DEFAULT_OFFSET_M,
                azimuths_deg=all_azimuths,
            )

            logger.info(
                "Инициализация калькулятора и индекса берега..."
            )

            calculator = PipelineFetchCalculator(
                working_crs=metadata["working_crs"],
                paths=paths,
                config=config,
            )

            staging_path = (
                temporary_dir / "fetch_staging.db"
            )

            staging_connection = sqlite3.connect(
                staging_path
            )

            try:
                total_rows = calculate_to_staging(
                    calculator,
                    normals_gdf,
                    staging_connection,
                )

                logger.info(
                    "Расчёт завершён: {} строк. "
                    "Начинается замена wind_fetches.",
                    total_rows,
                )

                point_ids = (
                    normals_gdf["point_id"]
                    .astype(int)
                    .tolist()
                )

                replace_fetches(
                    engine,
                    staging_connection,
                    metadata,
                    point_ids,
                    total_rows,
                )

            finally:
                staging_connection.close()

        logger.success(
            "Сохранено {} строк wind_fetches "
            "для {} точек; normal_source_id={}",
            total_rows,
            len(normals_gdf),
            metadata["id"],
        )

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()