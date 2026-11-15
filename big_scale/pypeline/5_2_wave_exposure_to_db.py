from __future__ import annotations

"""
Расчёт WEI-показателей, рангов R1-R4 и индекса WER.

Вход по умолчанию:
    big_scale/data/db/wave_activity.db

Выход по умолчанию:
    та же wave_activity.db

Расчёт выполняется базовым WaveExposureDatabaseBuilder
на согласованном снимке SQLite.

Если output == source:
    в одной транзакции обновляются только
    wave_exposure_index и метаданные этого расчёта.

Если output != source:
    сохраняется полная копия БД с рассчитанными индексами
    через SQLite Backup API.

Исходный файл SQLite не копируется через shutil.copy2.
Рабочие WAL/SHM-файлы вручную не удаляются.

Переменные окружения:
    WAVE_ACTIVITY_DB
    WAVE_INDEX_OUTPUT_DB
    WAVE_INDEX_OVERWRITE
    WAVE_INDEX_ALLOW_LEGACY
    WAVE_STORM_PERCENTILE
    WAVE_STORM_THRESHOLD
    WAVE_DT_SECONDS
    WAVE_LOG_LEVEL

WAVE_INDEX_OVERWRITE:
    разрешает заменить существующую отдельную выходную БД;
    не требуется для обновления индексов в исходной БД.

WAVE_INDEX_ALLOW_LEGACY:
    разрешает входную БД без таблиц resumable-checkpoint;
    полнота summary при этом всё равно проверяется.

macOS/Linux: используется fcntl.flock.
"""

import fcntl
import json
import math
import os
import sqlite3
import sys
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger


# ==================== КОРЕНЬ ПРОЕКТА ====================

def find_project_root() -> Path:
    directory = Path(__file__).resolve().parent

    for candidate in (directory, *directory.parents):
        if (
            candidate
            / "src/waves/services/wave_exposure_db_service.py"
        ).is_file():
            return candidate

    raise RuntimeError(
        "Не найден корень BreachTheBeach: отсутствует "
        "src/waves/services/wave_exposure_db_service.py"
    )


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "big_scale" / "data"


# До импорта NumPy/Pandas и проектных модулей.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"


INDEX_COLUMNS = (
    "point_id",
    "mean_cwef_wm",
    "e_storm_mjm",
    "storm_threshold_wm",
    "storm_percentile",
    "k_dir",
    "cv",
    "n_days",
    "n_storm_days",
    "top3_sectors",
    "r1",
    "r2",
    "r3",
    "r4",
    "wer",
)

METADATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS wave_exposure_build_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    created_at TEXT NOT NULL,
    config_json TEXT NOT NULL
)
"""


# ==================== НАСТРОЙКИ ====================

def env_path(
    name: str,
    default: Path,
) -> Path:
    raw = os.getenv(name, "").strip()

    path = Path(raw) if raw else default
    path = path.expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


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

    if not math.isfinite(value):
        raise ValueError(
            f"{name} должен быть конечным."
        )

    return value


def env_optional_float(
    name: str,
) -> float | None:
    raw = os.getenv(name, "").strip()

    if not raw:
        return None

    try:
        value = float(raw)

    except ValueError as error:
        raise ValueError(
            f"{name}: ожидается число, получено {raw!r}"
        ) from error

    if not math.isfinite(value):
        raise ValueError(
            f"{name} должен быть конечным."
        )

    return value


def env_bool(
    name: str,
    default: bool = False,
) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    raw = raw.strip().lower()

    if raw in {"1", "true", "yes", "on"}:
        return True

    if raw in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name}: ожидается true/false, получено {raw!r}"
    )


def configure_logging() -> None:
    logger.remove()

    logger.add(
        sys.stderr,
        level=os.getenv(
            "WAVE_LOG_LEVEL",
            "INFO",
        ).strip().upper(),
        diagnose=False,
        backtrace=False,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | {message}"
        ),
    )


# ==================== БЛОКИРОВКИ ====================

@contextmanager
def database_lock(path: Path):
    """
    Совместимо с блокировкой предыдущего скрипта волн:
        <database>.lock

    Lock-файл не удаляется.
    """
    lock_path = Path(f"{path}.lock")

    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )

        except BlockingIOError as error:
            raise RuntimeError(
                "БД уже используется другим этапом пайплайна:\n"
                f"{path}\n"
                f"Блокировка: {lock_path}"
            ) from error

        try:
            yield

        finally:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_UN,
            )


# ==================== SQLITE ====================

def connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(
        path.as_uri() + "?mode=ro",
        uri=True,
        timeout=60,
    )


def data_version(connection: sqlite3.Connection) -> int:
    return int(
        connection.execute(
            "PRAGMA data_version"
        ).fetchone()[0]
    )


def assert_source_unchanged(
    connection: sqlite3.Connection,
    expected_version: int,
) -> None:
    if data_version(connection) != expected_version:
        raise RuntimeError(
            "Исходная БД была изменена другим SQLite-соединением "
            "во время расчёта. Индексы не опубликованы."
        )


def create_snapshot(
    source_connection: sqlite3.Connection,
    destination: Path,
) -> None:
    logger.info(
        "Создание согласованного SQLite-снимка: {}",
        destination,
    )

    target = sqlite3.connect(
        destination,
        timeout=60,
    )

    try:
        source_connection.backup(
            target,
            pages=4096,
            sleep=0.1,
        )

    finally:
        target.close()


def check_integrity(connection: sqlite3.Connection) -> None:
    for row in connection.execute(
        "PRAGMA quick_check"
    ):
        if row[0] != "ok":
            raise ValueError(
                f"SQLite quick_check: {row[0]}"
            )

    foreign_key_error = connection.execute(
        "PRAGMA foreign_key_check"
    ).fetchone()

    if foreign_key_error is not None:
        raise ValueError(
            "Нарушены внешние ключи SQLite: "
            f"{foreign_key_error}"
        )


def validate_completed_activity(
    path: Path,
    *,
    allow_legacy: bool,
) -> int:
    connection = connect_readonly(path)

    try:
        check_integrity(connection)

        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table'"
            )
        }

        required = {
            "wave_points",
            "wave_activity",
            "wave_activity_summary",
            "wave_exposure_index",
        }

        missing_tables = required - tables

        if missing_tables:
            raise ValueError(
                "В БД отсутствуют таблицы: "
                f"{sorted(missing_tables)}"
            )

        point_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM wave_points"
            ).fetchone()[0]
        )

        if point_count == 0:
            raise ValueError(
                "wave_points пуста."
            )

        missing_summary = connection.execute(
            """
            SELECT p.id
            FROM wave_points AS p
            LEFT JOIN wave_activity_summary AS s
                ON s.point_id = p.id
            WHERE s.point_id IS NULL
            LIMIT 20
            """
        ).fetchall()

        if missing_summary:
            raise ValueError(
                "Не для всех точек рассчитана summary. "
                "Примеры ID: "
                f"{[row[0] for row in missing_summary]}"
            )

        checkpoint_tables = {
            "wave_build_meta",
            "wave_build_progress",
        }

        checkpoint_present = checkpoint_tables & tables

        if checkpoint_present and checkpoint_present != checkpoint_tables:
            raise ValueError(
                "Неполная структура resumable-checkpoint."
            )

        if checkpoint_present:
            meta = connection.execute(
                "SELECT status FROM wave_build_meta WHERE id=1"
            ).fetchone()

            if meta is None or meta[0] != "completed":
                raise ValueError(
                    "Расчёт волн не завершён: "
                    f"checkpoint status={None if meta is None else meta[0]!r}. "
                    "Сначала заверши расчёт wave_activity."
                )

            incomplete = connection.execute(
                """
                SELECT p.id
                FROM wave_points AS p
                LEFT JOIN wave_build_progress AS b
                    ON b.point_id = p.id
                WHERE b.point_id IS NULL
                   OR b.status <> 'done'
                LIMIT 20
                """
            ).fetchall()

            if incomplete:
                raise ValueError(
                    "В checkpoint есть незавершённые точки: "
                    f"{[row[0] for row in incomplete]}"
                )

        elif not allow_legacy:
            raise ValueError(
                "В БД нет resumable-checkpoint. "
                "Для осознанного расчёта по старой БД задай "
                "WAVE_INDEX_ALLOW_LEGACY=true."
            )

        else:
            logger.warning(
                "Разрешена legacy-БД без checkpoint: "
                "наличие summary проверено, но завершённость "
                "предыдущего расчёта независимо не подтверждается."
            )

        invalid_summary = connection.execute(
            """
            SELECT point_id
            FROM wave_activity_summary
            WHERE n_days < 0
               OR n_active_days < 0
               OR n_active_days > n_days
            LIMIT 20
            """
        ).fetchall()

        if invalid_summary:
            raise ValueError(
                "Некорректные счётчики в summary: "
                f"{[row[0] for row in invalid_summary]}"
            )

        stored_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM wave_activity"
            ).fetchone()[0]
        )

        logger.info(
            "Вход проверен: точек={}, "
            "сохранённых дневных записей={}",
            point_count,
            stored_rows,
        )

        return point_count

    finally:
        connection.close()


# ==================== МЕТАДАННЫЕ ИНДЕКСОВ ====================

def write_index_metadata(
    path: Path,
    config: dict,
) -> None:
    connection = sqlite3.connect(
        path,
        timeout=60,
    )

    try:
        with connection:
            connection.execute(
                METADATA_TABLE_SQL
            )

            connection.execute(
                """
                INSERT OR REPLACE INTO wave_exposure_build_meta (
                    id,
                    created_at,
                    config_json
                ) VALUES (1, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    json.dumps(
                        config,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                ),
            )

    finally:
        connection.close()


# ==================== ПУБЛИКАЦИЯ ====================

def publish_indices_in_place(
    source: Path,
    computed: Path,
    watcher: sqlite3.Connection,
    expected_version: int,
) -> None:
    """
    Обновляет только индексы и их метаданные.

    Дневные ряды, summary и wave_build_* не заменяются.
    """
    connection = sqlite3.connect(
        source.as_uri() + "?mode=rw",
        uri=True,
        timeout=60,
        isolation_level=None,
    )

    try:
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        connection.execute(
            "PRAGMA synchronous = FULL"
        )

        connection.execute(
            "ATTACH DATABASE ? AS computed",
            (
                computed.as_uri() + "?mode=ro",
            ),
        )

        connection.execute(
            "BEGIN IMMEDIATE"
        )

        try:
            # BEGIN IMMEDIATE уже запрещает другим
            # SQLite-writers начать конкурирующую запись.
            assert_source_unchanged(
                watcher,
                expected_version,
            )

            columns = ", ".join(
                INDEX_COLUMNS
            )

            connection.execute(
                "DELETE FROM main.wave_exposure_index"
            )

            connection.execute(
                f"""
                INSERT INTO main.wave_exposure_index (
                    {columns}
                )
                SELECT {columns}
                FROM computed.wave_exposure_index
                """
            )

            connection.execute(
                METADATA_TABLE_SQL
            )

            connection.execute(
                "DELETE FROM main.wave_exposure_build_meta"
            )

            connection.execute(
                """
                INSERT INTO main.wave_exposure_build_meta (
                    id,
                    created_at,
                    config_json
                )
                SELECT
                    id,
                    created_at,
                    config_json
                FROM computed.wave_exposure_build_meta
                """
            )

            connection.execute(
                "COMMIT"
            )

        except BaseException:
            if connection.in_transaction:
                connection.execute(
                    "ROLLBACK"
                )
            raise

    finally:
        connection.close()


def publish_full_database(
    computed: Path,
    output: Path,
) -> None:
    """
    Публикует полную БД через SQLite Backup API.

    Не удаляет прежний файл output и его WAL/SHM вручную.
    """
    source_connection = connect_readonly(
        computed
    )

    destination_connection = None

    try:
        destination_connection = sqlite3.connect(
            output,
            timeout=60,
        )

        destination_connection.execute(
            "PRAGMA synchronous = FULL"
        )

        source_connection.backup(
            destination_connection,
            pages=4096,
            sleep=0.1,
        )

    finally:
        if destination_connection is not None:
            destination_connection.close()

        source_connection.close()


# ==================== MAIN ====================

def main() -> None:
    configure_logging()

    source = env_path(
        "WAVE_ACTIVITY_DB",
        DATA_DIR / "db/wave_activity.db",
    )

    output = env_path(
        "WAVE_INDEX_OUTPUT_DB",
        source,
    )

    percentile = env_float(
        "WAVE_STORM_PERCENTILE",
        90.0,
    )

    threshold = env_optional_float(
        "WAVE_STORM_THRESHOLD"
    )

    dt_seconds = env_float(
        "WAVE_DT_SECONDS",
        86_400.0,
    )

    overwrite = env_bool(
        "WAVE_INDEX_OVERWRITE",
        False,
    )

    allow_legacy = env_bool(
        "WAVE_INDEX_ALLOW_LEGACY",
        False,
    )

    if not 0.0 < percentile < 100.0:
        raise ValueError(
            "WAVE_STORM_PERCENTILE должен быть в (0, 100)."
        )

    if threshold is not None and threshold < 0:
        raise ValueError(
            "WAVE_STORM_THRESHOLD должен быть >= 0."
        )

    if dt_seconds <= 0:
        raise ValueError(
            "WAVE_DT_SECONDS должен быть > 0."
        )

    if not source.is_file():
        raise FileNotFoundError(
            "БД волновой активности не найдена:\n"
            f"{source}\n"
            "Сначала заверши расчёт wave_activity."
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    in_place = source == output

    # Разные пути к одному inode через hard link:
    # это не независимая выходная БД.
    if (
        not in_place
        and output.exists()
        and source.samefile(output)
    ):
        raise ValueError(
            "Вход и выход являются hard links одного файла. "
            "Используй одинаковый путь для обновления индексов "
            "или действительно отдельную выходную БД."
        )

    if not in_place:
        reserved = {
            Path(f"{source}{suffix}")
            for suffix in (
                "-wal",
                "-shm",
                "-journal",
                ".lock",
            )
        }

        if output in reserved:
            raise ValueError(
                "Выход совпадает со служебным файлом входной БД."
            )

        if Path(f"{output}.lock") == source:
            raise ValueError(
                "Lock-файл выхода совпадает с входной БД."
            )

    from src.waves.services.wave_exposure_db_service import (
        WaveExposureDatabaseBuilder,
    )
    from src.waves.storage import (
        WaveActivityRepository,
    )

    logger.info(
        "Вход: {}",
        source,
    )

    logger.info(
        "Выход: {}",
        output,
    )

    logger.info(
        "Режим: {}",
        (
            "транзакционное обновление индексов в исходной БД"
            if in_place
            else "полная отдельная БД с индексами"
        ),
    )

    logger.info(
        "Штормовой percentile={}, threshold={}, dt={} с",
        percentile,
        threshold,
        dt_seconds,
    )

    # Единый порядок блокировок предотвращает взаимную
    # блокировку двух запусков с переставленными путями.
    lock_paths = sorted(
        {source, output},
        key=str,
    )

    with ExitStack() as stack:
        for path in lock_paths:
            stack.enter_context(
                database_lock(path)
            )

        if (
            not in_place
            and output.exists()
            and not overwrite
        ):
            raise FileExistsError(
                f"Выходная БД уже существует:\n{output}\n"
                "Укажи другой путь либо явно задай "
                "WAVE_INDEX_OVERWRITE=true."
            )

        watcher = connect_readonly(
            source
        )

        try:
            version = data_version(
                watcher
            )

            with TemporaryDirectory(
                prefix=f".{output.name}.wei-",
                dir=output.parent,
            ) as directory:
                temporary = (
                    Path(directory)
                    / "computed.db"
                )

                create_snapshot(
                    watcher,
                    temporary,
                )

                assert_source_unchanged(
                    watcher,
                    version,
                )

                expected_points = validate_completed_activity(
                    temporary,
                    allow_legacy=allow_legacy,
                )

                with WaveActivityRepository(
                    temporary
                ) as repository:
                    stats = WaveExposureDatabaseBuilder(
                        repository=repository,
                        storm_percentile=percentile,
                        storm_threshold=threshold,
                        dt_seconds=dt_seconds,
                    ).run()

                if stats.point_count != expected_points:
                    raise RuntimeError(
                        "Число рассчитанных индексов "
                        "не совпадает с числом входных точек: "
                        f"{stats.point_count} != {expected_points}"
                    )

                write_index_metadata(
                    temporary,
                    {
                        "source": str(source),
                        "builder": (
                            "WaveExposureDatabaseBuilder"
                        ),
                        "storm_percentile": percentile,
                        "storm_threshold_wm": threshold,
                        "dt_seconds": dt_seconds,
                        "point_count": stats.point_count,
                        "valid_index_count": (
                            stats.valid_index_count
                        ),
                        "activity_representation": (
                            "stored_positive_integer_CWEF_"
                            "plus_reconstructed_zeros"
                        ),
                    },
                )

                assert_source_unchanged(
                    watcher,
                    version,
                )

                if in_place:
                    publish_indices_in_place(
                        source,
                        temporary,
                        watcher,
                        version,
                    )

                else:
                    publish_full_database(
                        temporary,
                        output,
                    )

        finally:
            watcher.close()

    logger.success(
        "Готово: points={}, valid_WER={}, "
        "min_WER={}, max_WER={}, mean_WER={}, db={}",
        stats.point_count,
        stats.valid_index_count,
        stats.min_wer,
        stats.max_wer,
        stats.mean_wer,
        output,
    )

    if stats.valid_index_count != stats.point_count:
        logger.warning(
            "У {} точек WER не определён. "
            "Проверь n_days, компоненты индекса "
            "и результаты базового ранжирования.",
            stats.point_count - stats.valid_index_count,
        )


if __name__ == "__main__":
    main()
