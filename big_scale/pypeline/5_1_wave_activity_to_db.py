from __future__ import annotations

import fcntl
import json
import math
import multiprocessing as mp
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from loguru import logger


# ==================== КОРЕНЬ ПРОЕКТА ====================

def find_project_root() -> Path:
    directory = Path(__file__).resolve().parent

    for candidate in (directory, *directory.parents):
        if (
            candidate
            / "src/waves/services/resumable_wave_activity.py"
        ).is_file():
            return candidate

    raise RuntimeError(
        "Не найден корень BreachTheBeach: отсутствует "
        "src/waves/services/resumable_wave_activity.py"
    )


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "big_scale" / "data"


# До импорта NumPy/SciPy и проектных модулей.
# Эти значения наследуются рабочими процессами spawn.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"


# ==================== ПАРАМЕТРЫ ОКРУЖЕНИЯ ====================

def env_path(
    name: str,
    default: Path,
) -> Path:
    raw = os.getenv(name)

    path = Path(raw) if raw else default
    path = path.expanduser()

    # Относительные пути считаются от корня проекта,
    # а не от текущего рабочего каталога IDE.
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def env_date(
    name: str,
) -> date | None:
    raw = os.getenv(name, "").strip()

    if not raw:
        return None

    try:
        return date.fromisoformat(raw)

    except ValueError as error:
        raise ValueError(
            f"{name}: ожидается YYYY-MM-DD, получено {raw!r}"
        ) from error


def env_int(
    name: str,
    default: int,
    *,
    minimum: int = 1,
) -> int:
    raw = os.getenv(name, str(default)).strip()

    try:
        value = int(raw)

    except ValueError as error:
        raise ValueError(
            f"{name}: ожидается целое число, получено {raw!r}"
        ) from error

    if value < minimum:
        raise ValueError(
            f"{name} должен быть >= {minimum}, получено {value}"
        )

    return value


def env_optional_id(
    name: str,
) -> int | None:
    raw = os.getenv(name, "").strip()

    if not raw:
        return None

    try:
        value = int(raw)

    except ValueError as error:
        raise ValueError(
            f"{name}: ожидается положительный ID, "
            f"получено {raw!r}"
        ) from error

    if value < 1:
        raise ValueError(
            f"{name} должен быть >= 1"
        )

    return value


def env_float(
    name: str,
    default: float,
) -> float:
    raw = os.getenv(name, str(default)).strip()

    try:
        value = float(raw)

    except ValueError as error:
        raise ValueError(
            f"{name}: ожидается число, получено {raw!r}"
        ) from error

    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} должен быть конечным и > 0"
        )

    return value


def env_bool(
    name: str,
    default: bool = False,
) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    normalized = raw.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name}: ожидается true/false, получено {raw!r}"
    )


# ==================== ПРОВЕРКИ ФАЙЛОВ ====================

def same_file(
    first: Path,
    second: Path,
) -> bool:
    if first.resolve() == second.resolve():
        return True

    if first.exists() and second.exists():
        return first.samefile(second)

    return False


def require_file(
    path: Path,
    description: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} не найден:\n{path}"
        )


def assert_sqlite_quiet(
    path: Path,
) -> None:
    """
    Входные БД должны быть закрыты для записи.

    Не выполняем checkpoint автоматически:
    данный скрипт не должен изменять входные архивы.
    """
    for suffix in ("-wal", "-journal"):
        sidecar = Path(f"{path}{suffix}")

        try:
            size = sidecar.stat().st_size

        except FileNotFoundError:
            continue

        if size:
            raise RuntimeError(
                "У входной SQLite есть непустой журнал:\n"
                f"{sidecar}\n"
                "Останови запись, корректно закрой соединения "
                "и при необходимости выполни checkpoint "
                "в приложении, которое создаёт эту БД."
            )


def assert_output_safe(
    output: Path,
    inputs: list[Path],
) -> None:
    """
    Защищает не только сам output,
    но и создаваемые SQLite служебные файлы.
    """
    output_paths = [
        output,
        Path(f"{output}-wal"),
        Path(f"{output}-shm"),
        Path(f"{output}-journal"),
        Path(f"{output}.lock"),
    ]

    for candidate in output_paths:
        for source in inputs:
            if same_file(candidate, source):
                raise ValueError(
                    "Выходная БД или её служебный файл "
                    "совпадает с входным файлом:\n"
                    f"{candidate}"
                )


def read_manifest(
    path: Path,
) -> tuple[dict, list[Path]]:
    try:
        content = json.loads(
            path.read_text(encoding="utf-8")
        )

    except (OSError, ValueError) as error:
        raise ValueError(
            f"Не удалось прочитать манифест:\n{path}"
        ) from error

    if not isinstance(content, dict):
        raise ValueError(
            "Манифест должен содержать JSON-объект."
        )

    if content.get("schema_version") != 1:
        raise ValueError(
            "Неподдерживаемая версия манифеста: "
            f"{content.get('schema_version')!r}"
        )

    point_source_id = content.get("point_source_id")

    if (
        isinstance(point_source_id, bool)
        or not isinstance(point_source_id, int)
        or point_source_id < 1
    ):
        raise ValueError(
            "Некорректный point_source_id в манифесте."
        )

    points = content.get("points")
    records = content.get("tiles")

    if not isinstance(points, dict) or not points:
        raise ValueError(
            "В манифесте отсутствуют точки."
        )

    if not isinstance(records, list) or not records:
        raise ValueError(
            "В манифесте отсутствуют тайлы."
        )

    tile_ids = set()
    tile_paths = []

    for tile in records:
        if not isinstance(tile, dict):
            raise ValueError(
                "Некорректная запись тайла."
            )

        tile_id = tile.get("tile_id")
        filename = tile.get("file")

        if not isinstance(tile_id, str) or not tile_id:
            raise ValueError(
                "У тайла отсутствует tile_id."
            )

        if tile_id in tile_ids:
            raise ValueError(
                f"Повторный tile_id={tile_id!r}."
            )

        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"У тайла {tile_id!r} отсутствует file."
            )

        tile_ids.add(tile_id)

        # Поддерживаются:
        # - tiles/tile_x.nc;
        # - ../external_cache/tile_x.nc;
        # - абсолютные пути.
        tile_path = (
            path.parent / Path(filename)
        ).resolve()

        require_file(
            tile_path,
            f"Файл тайла {tile_id}",
        )

        tile_paths.append(tile_path)

    for point_id, metadata in points.items():
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Некорректные метаданные point_id={point_id}."
            )

        references = metadata.get("tile_ids")

        if (
            not isinstance(references, list)
            or not references
            or any(
                not isinstance(value, str)
                for value in references
            )
        ):
            raise ValueError(
                f"У point_id={point_id} отсутствуют "
                "корректные ссылки tile_ids."
            )

        unknown = set(references) - tile_ids

        if unknown:
            raise ValueError(
                f"point_id={point_id} ссылается "
                f"на неизвестные тайлы: {sorted(unknown)}"
            )

    return content, sorted(set(tile_paths))


# ==================== БЛОКИРОВКИ ====================

@contextmanager
def file_lock(
    path: Path,
    *,
    shared: bool,
    description: str,
):
    """
    macOS/Linux.

    Lock-файл не удаляется:
    flock снимается ОС при закрытии дескриптора.
    """
    mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX

    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(
                handle.fileno(),
                mode | fcntl.LOCK_NB,
            )

        except BlockingIOError as error:
            raise RuntimeError(
                f"{description}\nБлокировка: {path}"
            ) from error

        try:
            yield

        finally:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_UN,
            )


# ==================== ПОДГОТОВКА ТОЧЕК ====================

def load_and_validate_points(
    coastline: Path,
    archive,
    requested_normal_source_id: int | None,
    *,
    coordinate_tolerance_m: float,
):
    from pyproj import Geod
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from src.coastline.storage.models import (
        CoastlineNormalModel,
        CoastlineNormalSourceModel,
        WindFetchModel,
    )
    from src.waves.services.wave_activity_db_service import (
        load_wave_points,
        resolve_normal_source_id,
    )

    # Отдельное read-only соединение.
    # SessionLocal здесь намеренно не используется:
    # оно зависит от глобального DATABASE_URL.
    engine = create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(
            coastline.as_uri() + "?mode=ro",
            uri=True,
            timeout=60,
        ),
    )

    try:
        with Session(engine) as session:
            if requested_normal_source_id is None:
                # Не просто последний источник нормалей вообще,
                # а последний совместимый с текущим архивом.
                source_id = session.execute(
                    select(
                        CoastlineNormalSourceModel.id
                    )
                    .join(
                        CoastlineNormalModel,
                        CoastlineNormalModel.normal_source_id
                        == CoastlineNormalSourceModel.id,
                    )
                    .join(
                        WindFetchModel,
                        WindFetchModel.point_id
                        == CoastlineNormalModel.point_id,
                    )
                    .where(
                        CoastlineNormalSourceModel.point_source_id
                        == archive.point_source_id
                    )
                    .group_by(
                        CoastlineNormalSourceModel.id
                    )
                    .order_by(
                        CoastlineNormalSourceModel.id.desc()
                    )
                    .limit(1)
                ).scalar_one_or_none()

                if source_id is None:
                    raise ValueError(
                        "Нет источника нормалей с fetch "
                        "для point_source_id="
                        f"{archive.point_source_id} "
                        "из манифеста батиметрии."
                    )

                source_id = int(source_id)

            else:
                source_id = resolve_normal_source_id(
                    session,
                    requested_normal_source_id,
                )

            point_source_id = session.execute(
                select(
                    CoastlineNormalSourceModel.point_source_id
                )
                .where(
                    CoastlineNormalSourceModel.id
                    == source_id
                )
            ).scalar_one()

            if int(point_source_id) != archive.point_source_id:
                raise ValueError(
                    "Нормали и батиметрия относятся "
                    "к разным источникам точек:\n"
                    f"normal_source_id={source_id}, "
                    f"point_source_id={point_source_id};\n"
                    "bathymetry.point_source_id="
                    f"{archive.point_source_id}.\n"
                    "Обнови манифест батиметрии для нужного "
                    "набора точек или выбери совместимые нормали."
                )

            points = load_wave_points(
                session,
                source_id,
            )

            if not points:
                raise ValueError(
                    f"В normal_source_id={source_id} "
                    "нет точек."
                )

            point_ids = [
                point.point_id
                for point in points
            ]

            if len(point_ids) != len(set(point_ids)):
                raise ValueError(
                    "Источник нормалей содержит несколько "
                    "записей для одной береговой точки."
                )

            archive_ids = set(archive.point_ids)

            missing_archive = [
                point_id
                for point_id in point_ids
                if point_id not in archive_ids
            ]

            if missing_archive:
                raise ValueError(
                    "Не все точки нормалей присутствуют "
                    "в архиве батиметрии. "
                    f"Примеры ID: {missing_archive[:20]}.\n"
                    "Повторно запусти подготовку батиметрии: "
                    "готовые пространственные тайлы "
                    "будут использованы повторно."
                )

            geod = Geod(ellps="WGS84")

            for point in points:
                metadata = archive.point_metadata(
                    point.point_id
                )

                archive_lon = float(metadata["lon"])
                archive_lat = float(metadata["lat"])

                if not (
                    math.isfinite(point.lon)
                    and math.isfinite(point.lat)
                    and math.isfinite(
                        point.normal_azimuth_deg
                    )
                    and -180 <= point.lon <= 180
                    and -90 <= point.lat <= 90
                    and math.isfinite(archive_lon)
                    and math.isfinite(archive_lat)
                    and -180 <= archive_lon <= 180
                    and -90 <= archive_lat <= 90
                ):
                    raise ValueError(
                        "Некорректные координаты или нормаль "
                        f"для point_id={point.point_id}."
                    )

                _, _, distance = geod.inv(
                    point.lon,
                    point.lat,
                    archive_lon,
                    archive_lat,
                )

                if (
                    not math.isfinite(distance)
                    or distance > coordinate_tolerance_m
                ):
                    raise ValueError(
                        f"point_id={point.point_id}: координаты "
                        "в БД и манифесте различаются "
                        f"на {distance:.3f} м; "
                        "допуск "
                        f"{coordinate_tolerance_m:.3f} м."
                    )

            # Проверяем fetch до создания выходной БД.
            # Читаем курсором, чтобы не загружать все ORM-объекты.
            have_fetch = set()

            rows = session.execute(
                select(
                    WindFetchModel.point_id,
                    WindFetchModel.azimuth_deg,
                    WindFetchModel.fetch_length_m,
                )
                .join(
                    CoastlineNormalModel,
                    CoastlineNormalModel.point_id
                    == WindFetchModel.point_id,
                )
                .where(
                    CoastlineNormalModel.normal_source_id
                    == source_id
                )
            ).yield_per(10_000)

            for point_id, azimuth, length in rows:
                try:
                    azimuth = float(azimuth)
                    length = float(length)

                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Некорректный fetch: point_id={point_id}"
                    ) from error

                if (
                    not math.isfinite(azimuth)
                    or not math.isfinite(length)
                    or length <= 0
                ):
                    raise ValueError(
                        "Некорректный fetch: "
                        f"point_id={point_id}, "
                        f"azimuth={azimuth}, length={length}"
                    )

                have_fetch.add(int(point_id))

            missing_fetch = sorted(
                set(point_ids) - have_fetch
            )

            if missing_fetch:
                raise ValueError(
                    "Нет fetch для точек: "
                    f"{missing_fetch[:20]}"
                )

            return source_id, points

    finally:
        engine.dispose()


# ==================== MAIN ====================

def main() -> None:
    logger.remove()

    logger.add(
        sys.stderr,
        level=os.getenv(
            "WAVE_LOG_LEVEL",
            "INFO",
        ).strip().upper(),
        diagnose=False,
        backtrace=False,
    )

    weather = env_path(
        "WAVE_WEATHER_DB",
        DATA_DIR / "db/weather_compact.db",
    )

    coastline = env_path(
        "WAVE_COASTLINE_DB",
        DATA_DIR / "db/coastline.db",
    )

    manifest = env_path(
        "WAVE_BATHY_MANIFEST",
        DATA_DIR / "bathymetry/manifest.json",
    )

    output = env_path(
        "WAVE_OUTPUT_DB",
        DATA_DIR / "db/wave_activity.db",
    )

    start = env_date(
        "WAVE_START_DATE"
    )

    end = env_date(
        "WAVE_END_DATE"
    )

    if (
        start is not None
        and end is not None
        and start > end
    ):
        raise ValueError(
            "WAVE_START_DATE позже WAVE_END_DATE."
        )

    max_workers = env_int(
        "WAVE_MAX_WORKERS",
        4,
        minimum=0,
    )

    if max_workers == 0:
        max_workers = max(
            1,
            (os.cpu_count() or 1) - 1,
        )

    pool_points = env_int(
        "WAVE_POOL_BATCH_POINTS",
        128,
    )

    commit_rows = env_int(
        "WAVE_COMMIT_BATCH_SIZE",
        500,
    )

    commit_points = env_int(
        "WAVE_COMMIT_POINTS",
        4,
    )

    commit_seconds = env_float(
        "WAVE_COMMIT_SECONDS",
        10.0,
    )

    breaking = env_float(
        "WAVE_BREAKING_COEFF",
        0.55,
    )

    if breaking > 1:
        raise ValueError(
            "WAVE_BREAKING_COEFF должен быть <= 1."
        )

    coordinate_tolerance_m = env_float(
        "WAVE_COORDINATE_TOLERANCE_M",
        5.0,
    )

    requested_source_id = env_optional_id(
        "WAVE_NORMAL_SOURCE_ID"
    )

    overwrite = env_bool(
        "WAVE_OVERWRITE",
        False,
    )

    options = {
        "start_date": (
            start.isoformat()
            if start is not None
            else None
        ),
        "end_date": (
            end.isoformat()
            if end is not None
            else None
        ),
        "bathy_manifest_path": str(manifest),
        "bathy_n_steps": env_int(
            "WAVE_BATHY_STEPS",
            1000,
            minimum=2,
        ),
        "overwater_factor": env_float(
            "WAVE_OVERWATER_FACTOR",
            1.1,
        ),
        "breaking_coeff": breaking,
        "rho_water": env_float(
            "WAVE_RHO_WATER",
            1025.0,
        ),
        "g": env_float(
            "WAVE_GRAVITY",
            9.81,
        ),
    }

    require_file(
        weather,
        "Погодный архив",
    )

    require_file(
        coastline,
        "База береговой линии",
    )

    require_file(
        manifest,
        "Манифест батиметрии",
    )

    # Для возможных транзитивных импортов db.py.
    # Основная проверка ниже использует своё read-only соединение.
    from sqlalchemy.engine import URL

    os.environ["COASTLINE_DATABASE_URL"] = (
        URL.create(
            "sqlite",
            database=str(coastline),
        ).render_as_string(hide_password=False)
    )

    from src.bathymetry.archive import (
        BathymetryArchive,
    )
    from src.waves.services.resumable_wave_activity import (
        ALGORITHM_VERSION,
        CheckpointRepository,
        assert_inputs_unchanged,
        input_signature,
        run_incremental,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Хешируется и сам launcher.
    # Менять параметры производительности между продолжениями
    # следует через WAVE_*, не редактируя этот файл.
    code = sorted(
        (PROJECT_ROOT / "src").rglob("*.py")
    )

    code.append(
        Path(__file__).resolve()
    )

    lockfile = PROJECT_ROOT / "poetry.lock"

    if lockfile.is_file():
        code.append(lockfile)

    else:
        logger.warning(
            "poetry.lock отсутствует: зависимости "
            "не будут представлены этим файлом "
            "в fingerprint."
        )

    # Также фиксируем конфигурацию зависимостей, если она есть.
    pyproject = PROJECT_ROOT / "pyproject.toml"

    if pyproject.is_file():
        code.append(pyproject)

    # Проверяем до открытия выходного lock-файла.
    assert_output_safe(
        output,
        [
            weather,
            coastline,
            manifest,
            *code,
        ],
    )

    logger.info(
        "Корень проекта: {}",
        PROJECT_ROOT,
    )

    logger.info(
        "Погода: {}",
        weather,
    )

    logger.info(
        "Береговая БД: {}",
        coastline,
    )

    logger.info(
        "Батиметрия: {}",
        manifest,
    )

    logger.info(
        "Выходная БД: {}",
        output,
    )

    with file_lock(
        Path(f"{output}.lock"),
        shared=False,
        description=(
            "Другой расчёт уже использует "
            "эту выходную БД."
        ),
    ):
        with file_lock(
            manifest.parent / ".download.lock",
            shared=True,
            description=(
                "Сейчас обновляется архив батиметрии. "
                "Дождись завершения загрузчика."
            ),
        ):
            content, tile_paths = read_manifest(
                manifest
            )

            inputs = [
                weather,
                coastline,
                manifest,
                *tile_paths,
            ]

            assert_output_safe(
                output,
                inputs + code,
            )

            for database in (
                weather,
                coastline,
            ):
                assert_sqlite_quiet(
                    database
                )

            logger.info(
                "Манифест: point_source_id={}, "
                "точек={}, файлов тайлов={}",
                content["point_source_id"],
                len(content["points"]),
                len(tile_paths),
            )

            # Сначала фиксируем входы, затем читаем точки:
            # изменения во время подготовки будут обнаружены.
            signature, states = input_signature(
                inputs + code
            )

            for database in (
                weather,
                coastline,
            ):
                assert_sqlite_quiet(
                    database
                )

            archive = BathymetryArchive(
                manifest,
                coordinate_tolerance_m=(
                    coordinate_tolerance_m
                ),
            )

            source_id, points = load_and_validate_points(
                coastline,
                archive,
                requested_source_id,
                coordinate_tolerance_m=(
                    coordinate_tolerance_m
                ),
            )

            assert_inputs_unchanged(
                states
            )

            config = {
                "algorithm_version": ALGORITHM_VERSION,
                "normal_source_id": source_id,
                "point_count": len(points),
                "options": options,
                "inputs": signature,
            }

            if overwrite:
                logger.warning(
                    "WAVE_OVERWRITE=true: существующие "
                    "результаты в {} будут сброшены "
                    "сервисом checkpoint.",
                    output,
                )

            with CheckpointRepository(output) as repository:
                pending = repository.setup(
                    config,
                    points,
                    overwrite=overwrite,
                )

                logger.info(
                    "normal_source_id={}, "
                    "point_source_id={}",
                    source_id,
                    archive.point_source_id,
                )

                logger.info(
                    "Точек: {}; сохранено: {}; "
                    "осталось: {}; workers={}",
                    len(points),
                    len(points) - len(pending),
                    len(pending),
                    max_workers,
                )

                stats = run_incremental(
                    repository,
                    pending,
                    weather_path=weather,
                    coastline_path=coastline,
                    options=options,
                    input_states=states,
                    max_workers=max_workers,
                    pool_batch_points=pool_points,
                    commit_batch_size=commit_rows,
                    commit_points=commit_points,
                    commit_seconds=commit_seconds,
                )

            logger.info(
                "Итог: points={}, days={}, "
                "activity_rows={}, skipped_weather={}, "
                "land_days={}, missing_bathymetry_days={}, "
                "error_points={}",
                stats.point_count,
                stats.processed_day_count,
                stats.activity_count,
                stats.skipped_weather_rows,
                stats.land_sector_rows,
                stats.missing_bathymetry_rows,
                stats.error_point_count,
            )

            if stats.error_point_count:
                raise SystemExit(
                    "Некоторые точки завершились ошибкой. "
                    "Исправь причину и перезапусти "
                    "БЕЗ WAVE_OVERWRITE=true."
                )

            if stats.missing_bathymetry_rows:
                logger.warning(
                    "Сервис зафиксировал {} дней "
                    "с отсутствующей батиметрией. "
                    "В текущей физической реализации "
                    "такие случаи учитываются как нулевая "
                    "волновая мощность; проверь покрытие "
                    "и пропуски глубин перед использованием "
                    "итоговых статистик.",
                    stats.missing_bathymetry_rows,
                )

            logger.success(
                "Расчёт завершён: {}",
                output,
            )


if __name__ == "__main__":
    mp.freeze_support()
    main()