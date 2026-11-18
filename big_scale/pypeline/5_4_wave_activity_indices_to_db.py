from __future__ import annotations

"""Шаг 5.4: индексы по полному календарю исходного расчёта волн (оптимизировано).

Заменяет big_scale/pypeline/5_4_wave_activity_indices_to_db.py.
Не меняет src/ и входные БД. Использует ActivityIndexBuilder проекта:
формулы, периодизация, alongshore и тренды не менялись.

Что ускорено относительно прежней версии:
 * полный quick_check/foreign_key_check исходной БД выключен по умолчанию;
 * глобальные GROUP BY / WHERE-сканы wave_activity заменены проверкой схемы:
   STRICT, PRIMARY KEY(point_id, day), CHECK-ограничения и индексные границы дней;
   если гарантий схемы нет, запускается полный скан автоматически, с прогрессом;
 * проверки и builder используют один read-snapshot; watcher — отдельное соединение;
 * SHA-256 погодной БД сохранён (проверка происхождения), но с прогрессом;
 * NULL-безопасные проверки checkpoint;
 * итоговая проверка небольшой БД индексов остаётся полной.

Не ослаблено: полный календарь (n_days одинаков у всех точек), завершённый
checkpoint, происхождение погоды, неизменность входов до публикации.
Ограничение: скрипт не исправляет батиметрические пропуски исходного расчёта.
macOS/Linux: fcntl.flock. Входные БД во время запуска изменять нельзя.
Длительный read-snapshot может удерживать WAL/checkpoint. Возобновления
незаконченного расчёта нет. Подстановка reader локальна синхронному вызову.
"""

import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, timedelta, UTC
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
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

# True: расчёт разрешён, результат помечается метаданными о пропусках батиметрии.
ALLOW_BATHYMETRY_GAPS = True

# Полный аудит структуры исходной БД: может занять очень много времени.
FULL_SOURCE_CHECK = False

# Принудительный полный скан wave_activity даже при достаточных гарантиях схемы.
FORCE_FULL_CALENDAR_SCAN = False

# SHA-256 подтверждает, что погода та же, что при расчёте волн.
# Отключать только осознанно: происхождение календаря не будет проверено.
VERIFY_WEATHER_SHA256 = True

PROGRESS_SECONDS = 5.0
LOG_LEVEL = "INFO"


def resolve_path(path: Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_settings() -> None:
    require(isinstance(INTERVAL_YEARS, int) and not isinstance(INTERVAL_YEARS, bool)
            and INTERVAL_YEARS >= 1, "INTERVAL_YEARS должен быть целым >= 1.")
    require(math.isfinite(MAX_ANGLE_DEG) and 0 < MAX_ANGLE_DEG < 90,
            "MAX_ANGLE_DEG должен находиться в (0, 90).")
    require(math.isfinite(PROGRESS_SECONDS) and PROGRESS_SECONDS > 0,
            "PROGRESS_SECONDS должен быть > 0.")
    for value in (TREND_ONLY_COMPLETE_PERIODS, OVERWRITE_OUTPUT, ALLOW_BATHYMETRY_GAPS,
                  FULL_SOURCE_CHECK, FORCE_FULL_CALENDAR_SCAN, VERIFY_WEATHER_SHA256):
        require(isinstance(value, bool), "Булевы настройки должны быть bool.")


def assert_paths_safe(paths: list[Path]) -> None:
    for i, first in enumerate(paths):
        for second in paths[i + 1:]:
            same = first == second
            if first.exists() and second.exists():
                same = same or first.samefile(second)
            require(not same, f"Роли БД должны использовать разные файлы: {first}, {second}")
            for suffix in ("-wal", "-shm", "-journal", ".lock"):
                require(first != Path(f"{second}{suffix}") and second != Path(f"{first}{suffix}"),
                        "Одна БД совпадает со служебным файлом другой.")


@contextmanager
def database_lock(path: Path):
    # Совместимо с lock-файлами предыдущих этапов. Lock-файл не удаляется.
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
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=60)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA cache_size=-65536")
    return connection


def data_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA data_version").fetchone()[0])


def file_state(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def assert_unchanged(connection, path, version, state) -> None:
    if data_version(connection) != version or file_state(path) != state:
        raise RuntimeError(f"Входная БД изменилась, результат не опубликован: {path}")


@contextmanager
def progress_logging(connection: sqlite3.Connection, label: str, interval: float):
    # Обработчик вызывается внутри SQLite VM; возвращаем 0, чтобы не прерывать запрос.
    started = last = time.monotonic()

    def callback() -> int:
        nonlocal last
        now = time.monotonic()
        if now - last >= interval:
            logger.info("{}: выполняется, {:.0f} с", label, now - started)
            last = now
        return 0

    connection.set_progress_handler(callback, 100_000)
    try:
        yield
    finally:
        connection.set_progress_handler(None, 0)
        logger.info("{}: завершено за {:.1f} с", label, time.monotonic() - started)


def check_integrity(connection, label="SQLite", interval=PROGRESS_SECONDS) -> None:
    with progress_logging(connection, f"Аудит {label}", interval):
        for row in connection.execute("PRAGMA quick_check"):
            require(row[0] == "ok", f"{label} quick_check: {row[0]}")
        violation = connection.execute("PRAGMA foreign_key_check").fetchone()
        require(violation is None, f"{label}: нарушены внешние ключи: {violation}")


def read_checkpoint(connection, expected_schema: int) -> tuple[dict, int, int]:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    require(version == expected_schema, f"Версия wave_activity: {version}; ожидается {expected_schema}.")
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    needed = {"wave_points", "wave_activity", "wave_activity_summary",
              "wave_build_meta", "wave_build_progress"}
    require(needed <= tables, f"Нет обязательных таблиц: {sorted(needed - tables)}")
    meta = connection.execute("SELECT status, config_json FROM wave_build_meta WHERE id=1").fetchone()
    require(meta is not None and meta[0] == "completed", "Расчёт волн ещё не завершён.")
    config = json.loads(meta[1])
    require(isinstance(config, dict), "Некорректный checkpoint.config_json.")
    require(isinstance(config.get("options"), dict), "Нет checkpoint.options.")
    count = int(connection.execute("SELECT COUNT(*) FROM wave_points").fetchone()[0])
    require(count > 0, "wave_points пуста.")
    # NULL-безопасно: любое отсутствие данных считается ошибкой, а не пропускается.
    bad = connection.execute("""
        SELECT p.id FROM wave_points p
        LEFT JOIN wave_activity_summary s ON s.point_id=p.id
        LEFT JOIN wave_build_progress b ON b.point_id=p.id
        WHERE s.point_id IS NULL OR b.point_id IS NULL
           OR b.status IS NULL OR b.status<>'done'
           OR b.n_days IS NULL OR s.n_days IS NULL OR b.n_days<>s.n_days
        LIMIT 20
    """).fetchall()
    require(not bad, f"Незавершённые или несогласованные точки: {bad}")
    orphan = connection.execute("""
        SELECT s.point_id FROM wave_activity_summary s
        LEFT JOIN wave_points p ON p.id=s.point_id WHERE p.id IS NULL LIMIT 20
    """).fetchall()
    require(not orphan, f"Summary без точки: {orphan}")
    value = connection.execute("SELECT COALESCE(SUM(no_bathy),0) FROM wave_build_progress").fetchone()[0]
    return config, count, int(value)


def expected_weather_hash(config: dict, weather: Path) -> str:
    records = config.get("inputs")
    require(isinstance(records, list), "В checkpoint нет списка fingerprint входов.")
    matching = [record for record in records
                if isinstance(record, dict) and isinstance(record.get("path"), str)
                and resolve_path(Path(record["path"])) == weather]
    require(len(matching) == 1,
            "WEATHER_DB не найдена среди входов checkpoint. Укажи исходный путь погодной БД.")
    digest = matching[0].get("sha256")
    require(isinstance(digest, str) and len(digest) == 64
            and all(char in "0123456789abcdefABCDEF" for char in digest),
            "В checkpoint нет корректного SHA-256 погодной БД.")
    return digest.lower()


def assert_no_journal(path: Path) -> None:
    for suffix in ("-wal", "-journal"):
        journal = Path(f"{path}{suffix}")
        try:
            size = journal.stat().st_size
        except FileNotFoundError:
            continue
        if size:
            raise RuntimeError(f"Перед проверкой погодной БД закрой её и заверши checkpoint: {journal}. "
                               "Не удаляй журнал вручную.")


def sha256_file(path: Path, interval: float = PROGRESS_SECONDS) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    done, started = 0, time.monotonic()
    last = started
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            done += len(block)
            now = time.monotonic()
            if now - last >= interval:
                logger.info("SHA-256 погоды: {:.1f}% | {:.2f}/{:.2f} GiB | {:.0f} с",
                            100 * done / total if total else 100, done / 1024**3,
                            total / 1024**3, now - started)
                last = now
    return digest.hexdigest()


def choose_bounds(options: dict, weather_range: tuple[int, int], epoch: date) -> tuple[int, int]:
    start, end = weather_range
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


def weather_bounds(connection, weather: Path, config: dict, epoch: date) -> tuple[int, int, str | None]:
    assert_no_journal(weather)
    version, state = data_version(connection), file_state(weather)
    digest = None
    if VERIFY_WEATHER_SHA256:
        expected = expected_weather_hash(config, weather)
        logger.info("Проверка SHA-256 погодного архива: {}", weather)
        digest = sha256_file(weather)
        require(digest == expected, "Погодный архив отличается от использованного при расчёте волн.")
    else:
        logger.warning("Проверка SHA-256 погоды отключена: происхождение календаря не подтверждено.")
    logger.info("Определение полного календаря weather_days...")
    with progress_logging(connection, "Границы погодного календаря", PROGRESS_SECONDS):
        row = connection.execute("SELECT MIN(day), MAX(day) FROM weather_days").fetchone()
    require(row is not None and row[0] is not None and row[1] is not None, "Погодный архив пуст.")
    require(type(row[0]) is int and type(row[1]) is int, "Даты погоды должны быть целыми day offset.")
    assert_no_journal(weather)
    assert_unchanged(connection, weather, version, state)
    start, end = choose_bounds(config["options"], (row[0], row[1]), epoch)
    return start, end, digest


def _norm_sql(sql: str | None) -> str:
    sql = re.sub(r"/\*.*?\*/|--[^\n]*", "", sql or "", flags=re.S)
    return "".join(sql.lower().split())


def schema_guarantees(connection) -> tuple[bool, list[str]]:
    """Консервативная проверка фактической структуры и CHECK проектной схемы."""
    row = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='wave_activity'").fetchone()
    sql = _norm_sql(row[0] if row else None)
    columns = {r[1]: r for r in connection.execute("PRAGMA table_info(wave_activity)")}
    table = next((r for r in connection.execute("PRAGMA table_list")
                  if r[0] == "main" and r[1] == "wave_activity"), None)
    missing = []
    if table is None or not table[5]:
        missing.append("STRICT")
    if table is None or not table[4]:
        missing.append("WITHOUT ROWID")
    for name in ("point_id", "day", "wind_azimuth_deg", "cwef_wm"):
        col = columns.get(name)
        if col is None or col[2].upper() != "INTEGER" or not col[3]:
            missing.append(f"INTEGER NOT NULL {name}")
    pk = [r[1] for r in sorted(columns.values(), key=lambda r: r[5]) if r[5]]
    if pk != ["point_id", "day"]:
        missing.append("PRIMARY KEY(point_id,day)")
    for label, token in (("CHECK cwef_wm >= 1", "check(cwef_wm>=1)"),
                         ("CHECK wind 0..359", "check(wind_azimuth_degbetween0and359)")):
        if token not in sql:
            missing.append(label)
    if re.search(r"check\(daybetween0and[0-9]+\)", sql) is None:
        missing.append("CHECK day BETWEEN 0 AND integer")
    return not missing, missing


def activity_day_bounds(connection):
    values = []
    for order in ("ASC", "DESC"):
        sql = f"SELECT day FROM wave_activity ORDER BY day {order} LIMIT 1"
        plan = [r[3].upper() for r in connection.execute("EXPLAIN QUERY PLAN " + sql)]
        if any("TEMP B-TREE" in item for item in plan):
            logger.warning("Нет индексированного определения границ day: выполняется один полный проход.")
            with progress_logging(connection, "Границы дней wave_activity", PROGRESS_SECONDS):
                return connection.execute("SELECT MIN(day), MAX(day) FROM wave_activity").fetchone()
        row = connection.execute(sql).fetchone()
        values.append(None if row is None else row[0])
    return tuple(values)


def run_with_snapshot(builder_type, connection, source, computed, start, end):
    # В сервисе символ SourceActivityReader импортирован на уровне модуля.
    # Локально подставляем reader того же интерфейса, не меняя файлы src/.
    import importlib
    import numpy as np
    module = importlib.import_module("src.waves.activity_indices.service")
    original = module.SourceActivityReader
    class SnapshotReader(original):
        def __init__(self, path):
            require(Path(path).resolve() == source, "Неожиданный путь source reader.")
            self.path = source
            self._con = connection
        def __exit__(self, *args):
            # Соединением управляет main, а не builder.
            return None
        def day_bounds(self):
            # Календарные границы исходных строк уже проверены validate_calendar.
            return None
        def iter_activity(self, point_id):
            days, wind, cwef = super().iter_activity(point_id)
            require(np.all((days >= start) & (days <= end)), f"Дни вне календаря: point_id={point_id}")
            require(np.all(np.diff(days) > 0), f"Дубликаты/порядок дней: point_id={point_id}")
            require(np.isfinite(cwef).all() and np.all(cwef > 0), f"Некорректный CWEF: point_id={point_id}")
            require(np.isfinite(wind).all() and np.all((wind >= 0) & (wind <= 359)),
                    f"Некорректный ветер: point_id={point_id}")
            return days, wind, cwef
    with patch.object(module, "SourceActivityReader", SnapshotReader):
        return builder_type(source, computed, interval_years=INTERVAL_YEARS,
                            start_day=start, end_day=end, max_angle_deg=MAX_ANGLE_DEG,
                            trend_only_complete_periods=TREND_ONLY_COMPLETE_PERIODS).run()


def validate_calendar(connection, start: int, end: int, force_scan: bool, interval: float) -> str:
    length = end - start + 1
    bad = connection.execute("""
        SELECT point_id, n_days FROM wave_activity_summary
        WHERE n_days IS NULL OR n_days<>? OR n_active_days IS NULL
           OR n_active_days<0 OR n_active_days>n_days
        LIMIT 20
    """, (length,)).fetchall()
    require(not bad, f"Есть неполные календарные ряды; ожидалось {length} дней: {bad}. "
                     "Нельзя автоматически выдавать неизвестные погодные пропуски за нули.")
    guaranteed, missing = schema_guarantees(connection)
    row = activity_day_bounds(connection)
    if row[0] is not None:
        require(start <= row[0] and row[1] <= end, f"Дни активности {row} вне календаря {start}..{end}.")
    if guaranteed and not force_scan:
        logger.info("Глобальный скан wave_activity пропущен: типы, PK, CHECK и границы дней подтверждены.")
        return "schema_guarantees_and_index_bounds"
    reason = "FORCE_FULL_CALENDAR_SCAN=True" if force_scan else f"нет гарантий схемы: {missing}"
    logger.warning("Запускается полный скан wave_activity ({}). Это может быть долго.", reason)
    with progress_logging(connection, "Полный скан wave_activity", interval):
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
    return "full_scan"


def validate_result(path: Path, points: int, periods: int) -> None:
    connection = connect_readonly(path)
    try:
        check_integrity(connection, "результата")
        for table, expected in (("points", points), ("periods", periods),
                                ("point_trends", points), ("point_period_indices", points * periods)):
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            require(count == expected, f"{table}: получено {count}, ожидалось {expected}.")
    finally:
        connection.close()


def write_metadata(path, weather, no_bathy, calendar_check, source_audit, weather_digest) -> None:
    connection = sqlite3.connect(path, timeout=60)
    try:
        with connection:
            connection.executemany("INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)", [
                ("launcher_created_at", datetime.now(UTC).isoformat()),
                ("launcher_weather_db", str(weather)),
                ("launcher_weather_sha256_verified", "true" if weather_digest else "false"),
                ("calendar_source", "verified_weather_archive_and_checkpoint_limits"
                 if weather_digest else "weather_archive_not_hash_verified"),
                ("weather_calendar_policy", "require_full_calendar_at_each_point"),
                ("source_calendar_check", calendar_check),
                ("source_full_integrity_check", "true" if source_audit else "false"),
                ("source_missing_bathymetry_point_days", str(no_bathy)),
                ("source_quality", "bathymetry_gaps_present" if no_bathy else "no_reported_bathymetry_gaps"),
                ("activity_representation", "stored_positive_integer_CWEF"),
            ])
    finally:
        connection.close()


def publish_database(computed: Path, output: Path) -> None:
    source = connect_readonly(computed)
    target = None
    try:
        target = sqlite3.connect(output, timeout=60)
        target.execute("PRAGMA synchronous=FULL")
        started = last = time.monotonic()
        page_size = source.execute("PRAGMA page_size").fetchone()[0]
        def progress(status, remaining, total):
            nonlocal last
            now = time.monotonic()
            if now - last < PROGRESS_SECONDS and remaining != 0:
                return
            last = now
            done = max(0, total - remaining)
            logger.info("Публикация: {:.1f}% | {:.2f}/{:.2f} GiB | {:.0f} с | status={}",
                        100*done/total if total else 0, done*page_size/1024**3,
                        total*page_size/1024**3, now-started, status)
        source.backup(target, pages=4096, sleep=0.1, progress=progress)
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

    started = time.monotonic()
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
        # Отдельный watcher остаётся вне транзакции. Проверки и builder читают
        # одно соединение с закреплённым snapshot.
        wave_reader = connect_readonly(source)
        stack.callback(wave_reader.close)
        wave_reader.execute("BEGIN")
        wave_reader.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        assert_unchanged(wave_connection, source, wave_version, wave_state)
        if FULL_SOURCE_CHECK:
            check_integrity(wave_reader, "исходной wave_activity")
        else:
            logger.info("Полный аудит исходной БД отключён; включение: FULL_SOURCE_CHECK=True.")
        config, points, no_bathy = read_checkpoint(wave_reader, SOURCE_SCHEMA_VERSION)
        if no_bathy:
            message = (f"В исходной БД {no_bathy:,} точко-дней с отсутствующей батиметрией, "
                       "которые предыдущий расчёт учёл как нули. Этот этап их не восстанавливает.")
            if not ALLOW_BATHYMETRY_GAPS:
                raise ValueError(message)
            logger.warning("{} Результат будет помечен соответствующими метаданными.", message)
        weather_version, weather_state = data_version(weather_connection), file_state(weather)
        start, end, weather_digest = weather_bounds(weather_connection, weather, config, EPOCH)
        periods = build_periods(start, end, INTERVAL_YEARS)
        calendar_check = validate_calendar(wave_reader, start, end, FORCE_FULL_CALENDAR_SCAN, PROGRESS_SECONDS)
        assert_unchanged(wave_connection, source, wave_version, wave_state)
        assert_unchanged(weather_connection, weather, weather_version, weather_state)
        logger.info("Календарь: {}..{}; дней={}; точек={}; периодов={}",
                    EPOCH + timedelta(days=start), EPOCH + timedelta(days=end),
                    end - start + 1, points, len(periods))
        selected = sum(p.is_complete or not TREND_ONLY_COMPLETE_PERIODS for p in periods)
        if selected < 3:
            logger.warning("Для тренда выбрано {} периодов; модулю нужно не менее трёх.", selected)
        with TemporaryDirectory(prefix=f".{output.name}.build-", dir=output.parent) as directory:
            computed = Path(directory) / "indices.db"
            logger.info("Расчёт индексов: дневные данные читаются по одной точке.")
            stats = run_with_snapshot(ActivityIndexBuilder, wave_reader, source, computed, start, end)
            wave_reader.execute("ROLLBACK")
            write_metadata(computed, weather, no_bathy, calendar_check, FULL_SOURCE_CHECK, weather_digest)
            validate_result(computed, points, len(periods))
            assert_no_journal(weather)
            assert_unchanged(wave_connection, source, wave_version, wave_state)
            assert_unchanged(weather_connection, weather, weather_version, weather_state)
            publish_database(computed, output)
    logger.success("Готово: points={}, periods={}, days={}..{}, {:.1f} с, database={}",
                   stats.n_points, stats.n_periods, stats.start_day, stats.end_day,
                   time.monotonic() - started, output)
    if no_bathy:
        logger.warning("Индексы рассчитаны по БД с известными батиметрическими пропусками; это не исправленный физический расчёт.")


if __name__ == "__main__":
    main()
