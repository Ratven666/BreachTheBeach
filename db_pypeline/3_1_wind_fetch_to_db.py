from __future__ import annotations

"""
db_pypeline/3_1_wind_fetch_to_db.py

Расчёт длин разгона ветра и запись в таблицу wind_fetches.

В БД сохраняются только направления в пределах морского сектора
(±90° от нормали точки). Направления, попадающие в сушевой сектор,
отбрасываются — они лишены физического смысла для задачи разгона волн.

Опционально: для точек на краях отрезка береговой линии можно
передать расширенную (дополненную) линию через EXTENDED_COASTLINE_PATH,
чтобы исключить артефакты «открытых торцов».

Последовательность пайплайна:
    python db_pypeline/1_1_import_coastline_points.py
    python db_pypeline/2_1_normals_to_db.py
    python db_pypeline/3_1_wind_fetch_to_db.py
"""

import os
import sys
import tempfile
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

# Шаг разбиения морского полукруга, градусы
AZIMUTH_STEP_DEG: float = 1.0

# Максимальная длина луча при отсутствии пересечения с берегом
DEFAULT_FETCH_M: float = 100_000.0

# Смещение стартовой точки от береговой линии в сторону моря
DEFAULT_OFFSET_M: float = 1.0

SHOW_PROGRESS: bool = True

# ------------------------------------------------------------------
# Расширенная береговая линия (опционально).
#
# Если задана — используется как ОСНОВНАЯ линия для трассировки лучей
# вместо той, что записана в CoastlineSourceModel. Это позволяет
# минимизировать артефакты «открытого торца» для точек на краях
# исходного отрезка: расширенная линия перекрывает эти торцы.
#
# Оригинальная линия из CoastlineSourceModel в этом режиме передаётся
# как other_coastline_path и участвует в построении объединённой
# геометрии (combined_gdf).
#
# Если None — используется поведение по умолчанию (линия из БД).
# ------------------------------------------------------------------

# EXTENDED_COASTLINE_PATH: str | None = None
EXTENDED_COASTLINE_PATH: str | None = "data/coastline/BlackSeaArea.geojson"

# Пример:
# EXTENDED_COASTLINE_PATH = "data/coastline/coastline_extended.geojson"


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------

def setup_env() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if "COASTLINE_DATABASE_URL" not in os.environ:
        db_url = f"sqlite:///{DATABASE_PATH.resolve().as_posix()}"
        os.environ["COASTLINE_DATABASE_URL"] = db_url
        logger.debug(f"COASTLINE_DATABASE_URL → {db_url}")


def _angular_distance(a: float, b: float) -> float:
    """Минимальный угол между двумя азимутами [0, 180]."""
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def main() -> None:
    setup_env()

    from sqlalchemy import delete, select

    from src.coastline.storage import db as database
    from src.coastline.storage.CoastlineNormalRepository import (
        CoastlineNormalRepository,
    )
    from src.coastline.storage.models import (
        Base,
        CoastlineNormalSourceModel,
        CoastlineSourceModel,
        WindFetchModel,
    )
    from src.wind_fetch.SequentialMultiDirectionFetchCalculator import (
        SequentialMultiDirectionFetchCalculator,
    )
    from src.wind_fetch.WindFetchConfig import WindFetchConfig
    from src.wind_fetch.models import WindFetchPaths

    Base.metadata.create_all(database.engine)
    logger.info("DB schema ready")

    # ------------------------------------------------------------------
    # 1. Загружаем метаданные набора нормалей
    # ------------------------------------------------------------------
    with database.SessionLocal() as session:
        if NORMAL_SOURCE_ID is None:
            normal_source = (
                session.execute(
                    select(CoastlineNormalSourceModel).order_by(
                        CoastlineNormalSourceModel.id.desc()
                    )
                )
                .scalars()
                .first()
            )
        else:
            normal_source = session.get(
                CoastlineNormalSourceModel, NORMAL_SOURCE_ID
            )

        if normal_source is None:
            raise ValueError(
                "Набор нормалей не найден. "
                "Сначала запустите db_pypeline/2_1_normals_to_db.py"
            )

        point_source = session.get(
            CoastlineSourceModel, normal_source.point_source_id
        )
        if point_source is None:
            raise ValueError(
                f"CoastlineSource id={normal_source.point_source_id} not found"
            )
        if not point_source.main_geojson_path:
            raise ValueError(
                "У источника точек не задан main_geojson_path. "
                "Перезапустите 1_1_import_coastline_points.py."
            )

        db_main_path = point_source.main_geojson_path
        db_other_path = point_source.other_geojson_path  # может быть None
        normal_source_id = normal_source.id

        # ------------------------------------------------------------------
        # 2. Загружаем GDF нормалей (load_as_gdf возвращает EPSG:4326)
        # ------------------------------------------------------------------
        repo = CoastlineNormalRepository(session)
        normals_gdf = repo.load_as_gdf(normal_source_id)

    if normals_gdf.empty:
        raise ValueError(
            f"Normal source id={normal_source_id} не содержит нормалей"
        )

    # Санитарная проверка CRS
    if normals_gdf.crs is None:
        normals_gdf = normals_gdf.set_crs("EPSG:4326")
        logger.warning("load_as_gdf вернул GDF без CRS — принудительно EPSG:4326")
    elif str(normals_gdf.crs).upper() != "EPSG:4326":
        logger.warning(
            f"load_as_gdf вернул CRS={normals_gdf.crs}, ожидалась EPSG:4326. "
            "Перепроецируем."
        )
        normals_gdf = normals_gdf.to_crs("EPSG:4326")

    logger.info(
        f"Загружено нормалей: {len(normals_gdf)}, "
        f"CRS={normals_gdf.crs}, "
        f"normal_source id={normal_source_id}"
    )

    # ------------------------------------------------------------------
    # 3. Определяем пути береговых линий для калькулятора.
    #
    # Если задана расширенная линия:
    #   main_coastline_path  = EXTENDED_COASTLINE_PATH  (расширенная)
    #   other_coastline_path = db_main_path              (оригинальная)
    #
    # Это позволяет CoastlineDataset объединить обе линии в combined_gdf,
    # закрыв «открытые торцы» расширенной геометрией.
    #
    # Если расширенная линия не задана — стандартное поведение.
    # ------------------------------------------------------------------
    if EXTENDED_COASTLINE_PATH is not None:
        extended_path = str(Path(EXTENDED_COASTLINE_PATH).resolve())
        if not Path(extended_path).exists():
            raise FileNotFoundError(
                f"EXTENDED_COASTLINE_PATH не найден: {extended_path}"
            )
        main_coastline_path = extended_path
        other_coastline_path = db_main_path
        logger.info(
            f"Режим расширенной береговой линии:\n"
            f"  main  (расширенная) = {main_coastline_path}\n"
            f"  other (оригинальная) = {other_coastline_path}"
        )
    else:
        main_coastline_path = db_main_path
        other_coastline_path = db_other_path
        logger.info(
            f"Стандартный режим береговой линии:\n"
            f"  main  = {main_coastline_path}\n"
            f"  other = {other_coastline_path}"
        )

    # ------------------------------------------------------------------
    # 4. Запускаем калькулятор (полный круг 0–359° с нужным шагом)
    # ------------------------------------------------------------------
    all_azimuths = [
        round(i * AZIMUTH_STEP_DEG, 6)
        for i in range(int(360.0 / AZIMUTH_STEP_DEG))
    ]

    with tempfile.TemporaryDirectory(prefix="breachthebeach_fetch_") as tmp:
        normals_path = Path(tmp) / "normals.geojson"
        normals_gdf.to_file(str(normals_path), driver="GeoJSON")

        paths = WindFetchPaths(
            main_coastline_path=main_coastline_path,
            other_coastline_path=other_coastline_path,
            points_with_normals_path=str(normals_path),
        )

        config = WindFetchConfig(
            default_fetch_m=DEFAULT_FETCH_M,
            default_offset_m=DEFAULT_OFFSET_M,
            azimuths_deg=all_azimuths,
        )

        calculator = SequentialMultiDirectionFetchCalculator(
            paths=paths,
            config=config,
        )

        logger.info(
            f"Запуск расчёта: точек={len(normals_gdf)}, "
            f"направлений={len(all_azimuths)}, "
            f"шаг={AZIMUTH_STEP_DEG}°, "
            f"default_fetch_m={DEFAULT_FETCH_M}, "
            f"default_offset_m={DEFAULT_OFFSET_M}"
        )

        results = calculator.calculate(show_progress=SHOW_PROGRESS)

    logger.info(f"Расчёт завершён: всего результатов = {len(results)}")

    # ------------------------------------------------------------------
    # 5. Фильтруем: только морской сектор ±90° от нормали.
    #    Пропускаем skipped_by_land_sector и угол > 90°.
    # ------------------------------------------------------------------
    insert_rows: list[dict] = []

    for r in results:
        if r.skipped_by_land_sector:
            continue
        if _angular_distance(r.azimuth_deg, r.normal_azimuth_deg) > 90.0:
            continue
        insert_rows.append(
            {
                "point_id":       r.point_id,
                "azimuth_deg":    r.azimuth_deg,
                "fetch_length_m": round(r.fetch_length_m),
            }
        )

    logger.info(
        f"После фильтрации морского сектора: {len(insert_rows)} строк "
        f"(отброшено {len(results) - len(insert_rows)})"
    )

    if not insert_rows:
        logger.warning("Нет данных для записи в wind_fetches — проверьте нормали.")
        return

    # ------------------------------------------------------------------
    # 6. Атомарная замена: удаляем старые записи, пишем новые
    # ------------------------------------------------------------------
    point_ids_to_replace = {row["point_id"] for row in insert_rows}

    with database.SessionLocal.begin() as session:
        session.execute(
            delete(WindFetchModel).where(
                WindFetchModel.point_id.in_(point_ids_to_replace)
            )
        )
        session.execute(
            WindFetchModel.__table__.insert(),
            insert_rows,
        )

    logger.success(
        f"Записано {len(insert_rows)} строк в wind_fetches "
        f"для {len(point_ids_to_replace)} точек "
        f"(normal_source id={normal_source_id})"
    )


if __name__ == "__main__":
    main()