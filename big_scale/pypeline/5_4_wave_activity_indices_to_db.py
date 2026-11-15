from __future__ import annotations

"""Шаг 5.4: индексы по полному календарю исходного расчёта волн.

Заменяет big_scale/pypeline/5_4_wave_activity_indices_to_db.py.
Не меняет src/ и входные БД. Настройки находятся ниже.
Использует ActivityIndexBuilder базового проекта.
"""

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import sys
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, timedelta, UTC
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger


def find_project_root() -> Path:
    directory = Path(__file__).resolve().parent
    for candidate in (directory, *directory.parents):
        if (candidate / "src/waves/activity_indices/service.py").is_file():
            return candidate
    raise RuntimeError("Не найден корень проекта BreachTheBeach.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

# ==================== НАСТРОЙКИ ====================

DATA_DIR = PROJECT_ROOT / "big_scale" / "data"
ACTIVITY_DB = DATA_DIR / "db/wave_activity.db"
WEATHER_DB = DATA_DIR / "db/weather_compact.db"
INDICES_DB = DATA_DIR / "db/wave_activity_indices.db"

INTERVAL_YEARS = 5
MAX_ANGLE_DEG = 85.0
TREND_ONLY_COMPLETE_PERIODS = True
OVERWRITE_OUTPUT = False

# True: расчёт по уже сохранённой wave_activity разрешён,
# но результат явно помечается как содержащий батиметрические пропуски.
# False: при no_bathy > 0 остановиться до вычисления индексов.
ALLOW_BATHYMETRY_GAPS = True

LOG_LEVEL = "INFO"

# Границы вручную не задаются. Используется полный календарь WEATHER_DB,
# ограниченный start_date/end_date из checkpoint, если они были заданы.
# Нельзя использовать MIN/MAX(day) из разреженной wave_activity.


def resolve_path(path: Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_settings() -> None:
    require(
        isinstance(INTERVAL_YEARS, int)
        and not isinstance(INTERVAL_YEARS, bool)
        and INTERVAL_YEARS >= 1,
        "INTERVAL_YEARS должен быть целым >= 1.",
    )
    require(
        math.isfinite(MAX_ANGLE_DEG) and 0 < MAX_ANGLE_DEG < 90,
        "MAX_ANGLE_DEG должен находиться в (0, 90).",
    )
    for value in (TREND_ONLY_COMPLETE_PERIODS, OVERWRITE_OUTPUT, ALLOW_BATHYMETRY_GAPS):
        require(isinstance(value, bool), "Булевы настройки должны быть bool.")


def assert_paths_safe(paths: list[Path]) -> None:
    for i, first in enumerate(paths):
        for second in paths[i + 1:]:
            same = first == second
            if first.exists() and second.exists():
                same = same or first.samefile(second)
            require(not same, f"Роли БД должны использовать разные файлы: {first}, {second}")
            for suffix in ("-wal", "-shm", "-journal", ".lock"):
                require(
                    first != Path(f"{second}{suffix}")
                    and second != Path(f"{first}{suffix}"),
                    "Одна БД совпадает со служебным файлом другой.",
                )


@contextmanager
def database_lock(path: Path):
    # Совместимо с lock-файлами предыдущих этапов. Не удалять lock-файл.
    with Path(f"{path}.lock").open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"БД используется другим этапом: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=60)


def data_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA data_version").fetchone()[0])


def file_state(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def assert_unchanged(connection, path, version, state) -> None:
    if data_version(connection) != version or file_state(path) != state:
        raise RuntimeError(f"Входная БД изменилась, результат не опубликован: {path}")


def check_integrity(connection) -> None:
    for row in connection.execute("PRAGMA quick_check"):
        require(row[0] == "ok", f"SQLite quick_check: {row[0]}")
    violation = connection.execute("PRAGMA foreign_key_check").fetchone()
    require(violation is None, f"Нарушены внешние ключи: {violation}")


def read_checkpoint(connection, expected_schema: int) -> tuple[dict, int, int]:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    require(version == expected_schema, f"Версия wave_activity: {version}; ожидается {expected_schema}.")
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    needed = {
        "wave_points", "wave_activity", "wave_activity_summary",
        "wave_build_meta", "wave_build_progress",
    }
    require(needed <= tables, f"Нет обязательных таблиц: {sorted(needed - tables)}")
    meta = connection.execute(
        "SELECT status, config_json FROM wave_build_meta WHERE id=1"
    ).fetchone()
    require(meta is not None and meta[0] == "completed", "Расчёт волн ещё не завершён.")
    config = json.loads(meta[1])
    require(isinstance(config, dict), "Некорректный checkpoint.config_json.")
    require(isinstance(config.get("options"), dict), "Нет checkpoint.options.")
    count = int(connection.execute("SELECT COUNT(*) FROM wave_points").fetchone()[0])
    require(count > 0, "wave_points пуста.")
    bad = connection.execute("""
        SELECT p.id FROM wave_points p
        LEFT JOIN wave_activity_summary s ON s.point_id=p.id
        LEFT JOIN wave_build_progress b ON b.point_id=p.id
        WHERE s.point_id IS NULL OR b.point_id IS NULL OR b.status<>'done'
           OR b.n_days<>s.n_days
        LIMIT 20
    """).fetchall()
    require(not bad, f"Незавершённые или несогласованные точки: {bad}")
    no_bathy = int(connection.execute(
        "SELECT COALESCE(SUM(no_bathy),0) FROM wave_build_progress"
    ).fetchone()[0])
    return config, count, no_bathy


def expected_weather_hash(config: dict, weather: Path) -> str:
    records = config.get("inputs")
    require(isinstance(records, list), "В checkpoint нет списка fingerprint входов.")
    matching = [
        record for record in records
        if isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and resolve_path(Path(record["path"])) == weather
    ]
    require(
        len(matching) == 1,
        "WEATHER_DB не найдена среди входов checkpoint. Укажи исходный путь погодной БД.",
    )
    digest = matching[0].get("sha256")
    require(
        isinstance(digest, str) and len(digest) == 64
        and all(char in "0123456789abcdefABCDEF" for char in digest),
        "В checkpoint нет корректного SHA-256 погодной БД.",
    )
    return digest.lower()


def assert_no_journal(path: Path) -> None:
    for suffix in ("-wal", "-journal"):
        journal = Path(f"{path}{suffix}")
        try:
            size = journal.stat().st_size
        except FileNotFoundError:
            continue
        if size:
            raise RuntimeError(
                f"Перед проверкой SHA-256 закрой погодную БД и заверши checkpoint: {journal}. "
                "Не удаляй журнал вручную."
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_bounds(options: dict, weather_bounds: tuple[int, int], epoch: date) -> tuple[int, int]:
    start, end = weather_bounds
    for key in ("start_date", "end_date"):
        raw = options.get(key)
        if raw is None:
            continue
        require(isinstance(raw, str), f"Некорректный checkpoint.options.{key}")
        value = (date.fromisoformat(raw) - epoch).days
        if key == "start_date":
            start = max(start, value)
        else:
            end = min(end, value)
    require(start <= end, "Ограничения checkpoint не пересекаются с календарём погоды.")
    return start, end


def weather_bounds(connection, weather, config, epoch) -> tuple[int, int]:
    assert_no_journal(weather)
    version, state = data_version(connection), file_state(weather)
    expected = expected_weather_hash(config, weather)
    logger.info("Проверка SHA-256 погодного архива: {}", weather)
    require(
        sha256_file(weather) == expected,
        "Погодный архив отличается от использованного при расчёте волн.",
    )
    logger.info("Чтение полного календарного диапазона weather_days...")
    row = connection.execute("SELECT MIN(day), MAX(day) FROM weather_days").fetchone()
    require(row is not None and row[0] is not None and row[1] is not None, "Погодный архив пуст.")
    require(type(row[0]) is int and type(row[1]) is int, "Даты погоды должны быть целыми day offset.")
    assert_no_journal(weather)
    assert_unchanged(connection, weather, version, state)
    return choose_bounds(config["options"], (row[0], row[1]), epoch)


def validate_calendar(connection, start: int, end: int) -> None:
    length = end - start + 1
    bad = connection.execute("""
        SELECT point_id, n_days FROM wave_activity_summary
        WHERE n_days<>? OR n_active_days<0 OR n_active_days>n_days
        LIMIT 20
    """, (length,)).fetchall()
    require(
        not bad,
        f"Есть неполные календарные ряды; ожидалось {length} дней: {bad}. "
        "Нельзя автоматически выдавать неизвестные погодные пропуски за нули.",
    )
    bad = connection.execute("""
        SELECT point_id, day FROM wave_activity
        WHERE typeof(day)<>'integer' OR day<? OR day>?
           OR typeof(cwef_wm)<>'integer' OR cwef_wm<=0
           OR typeof(wind_azimuth_deg)<>'integer'
           OR wind_azimuth_deg NOT BETWEEN 0 AND 359
        LIMIT 20
    """, (start, end)).fetchall()
    require(not bad, f"Некорректные дневные записи: {bad}")
    bad = connection.execute("""
        SELECT point_id, day FROM wave_activity
        GROUP BY point_id, day HAVING COUNT(*)>1 LIMIT 20
    """).fetchall()
    require(not bad, f"Повторные записи точки за одну дату: {bad}")


def validate_result(path: Path, points: int, periods: int) -> None:
    connection = connect_readonly(path)
    try:
        check_integrity(connection)
        for table, expected in (
            ("points", points), ("periods", periods),
            ("point_trends", points), ("point_period_indices", points * periods),
        ):
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            require(count == expected, f"{table}: получено {count}, ожидалось {expected}.")
    finally:
        connection.close()


def write_metadata(path, weather, no_bathy) -> None:
    connection = sqlite3.connect(path, timeout=60)
    try:
        with connection:
            connection.executemany(
                "INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)",
                [
                    ("launcher_created_at", datetime.now(UTC).isoformat()),
                    ("launcher_weather_db", str(weather)),
                    ("calendar_source", "verified_weather_archive_and_checkpoint_limits"),
                    ("weather_calendar_policy", "require_full_calendar_at_each_point"),
                    ("source_missing_bathymetry_point_days", str(no_bathy)),
                    ("source_quality", "bathymetry_gaps_present" if no_bathy else "no_reported_bathymetry_gaps"),
                    ("activity_representation", "stored_positive_integer_CWEF"),
                ],
            )
    finally:
        connection.close()


def publish_database(computed: Path, output: Path) -> None:
    source = connect_readonly(computed)
    target = None
    try:
        target = sqlite3.connect(output, timeout=60)
        target.execute("PRAGMA synchronous=FULL")
        source.backup(target, pages=4096, sleep=0.1)
    finally:
        if target is not None:
            target.close()
        source.close()


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL, diagnose=False, backtrace=False)
    validate_settings()
    source, weather, output = map(resolve_path, (ACTIVITY_DB, WEATHER_DB, INDICES_DB))
    for path in (source, weather):
        if not path.is_file():
            raise FileNotFoundError(path)
    assert_paths_safe([source, weather, output])
    output.parent.mkdir(parents=True, exist_ok=True)

    from src.waves.activity_indices import ActivityIndexBuilder
    from src.waves.activity_indices.constants import EPOCH, SOURCE_SCHEMA_VERSION
    from src.waves.activity_indices.periods import build_periods

    logger.info("Волновая активность: {}", source)
    logger.info("Погодный календарь: {}", weather)
    logger.info("Выход: {}", output)
    with ExitStack() as stack:
        for path in sorted({source, weather, output}, key=str):
            stack.enter_context(database_lock(path))
        if output.exists() and not OVERWRITE_OUTPUT:
            raise FileExistsError(f"Выход уже существует: {output}. Для пересчёта задай OVERWRITE_OUTPUT=True.")
        wave_connection = connect_readonly(source)
        weather_connection = connect_readonly(weather)
        stack.callback(wave_connection.close)
        stack.callback(weather_connection.close)
        wave_version, wave_state = data_version(wave_connection), file_state(source)
        check_integrity(wave_connection)
        config, points, no_bathy = read_checkpoint(wave_connection, SOURCE_SCHEMA_VERSION)
        if no_bathy:
            message = (
                f"В исходной БД {no_bathy:,} точко-дней с отсутствующей батиметрией, "
                "которые предыдущий расчёт учёл как нули. Этот этап их не восстанавливает."
            )
            if not ALLOW_BATHYMETRY_GAPS:
                raise ValueError(message)
            logger.warning("{} Результат будет помечен соответствующими метаданными.", message)
        start, end = weather_bounds(weather_connection, weather, config, EPOCH)
        periods = build_periods(start, end, INTERVAL_YEARS)
        validate_calendar(wave_connection, start, end)
        assert_unchanged(wave_connection, source, wave_version, wave_state)
        logger.info(
            "Все даты исходного расчёта: {}..{}; дней={}; точек={}; периодов={}",
            EPOCH + timedelta(days=start), EPOCH + timedelta(days=end),
            end - start + 1, points, len(periods),
        )
        selected = sum(p.is_complete or not TREND_ONLY_COMPLETE_PERIODS for p in periods)
        if selected < 3:
            logger.warning("Для тренда выбрано {} периодов; модулю нужно не менее трёх.", selected)
        with TemporaryDirectory(prefix=f".{output.name}.build-", dir=output.parent) as directory:
            computed = Path(directory) / "indices.db"
            stats = ActivityIndexBuilder(
                source, computed, interval_years=INTERVAL_YEARS,
                start_day=start, end_day=end, max_angle_deg=MAX_ANGLE_DEG,
                trend_only_complete_periods=TREND_ONLY_COMPLETE_PERIODS,
            ).run()
            validate_result(computed, points, len(periods))
            write_metadata(computed, weather, no_bathy)
            assert_unchanged(wave_connection, source, wave_version, wave_state)
            publish_database(computed, output)
    logger.success(
        "Готово: points={}, periods={}, days={}..{}, database={}",
        stats.n_points, stats.n_periods, stats.start_day, stats.end_day, output,
    )
    if no_bathy:
        logger.warning("Индексы рассчитаны по БД с известными батиметрическими пропусками; это не исправленный физический расчёт.")


if __name__ == "__main__":
    main()
