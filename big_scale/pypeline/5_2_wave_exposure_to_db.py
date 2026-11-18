from __future__ import annotations

"""Потоковый расчёт WEI/R1–R4/WER, macOS/Linux.

Полная замена big_scale/pypeline/5_2_wave_exposure_to_db.py.
Формулы и ранжирование выполняет проектный WaveExposureDatabaseBuilder.
Адаптер меняет только чтение: дневные данные одной точки вместо всей БД.

По умолчанию: обновление wave_activity.db БЕЗ её полной копии.
Расчёт читает согласованный SQLite read-snapshot. Во временной маленькой БД
хранятся только point_id, индексы и метаданные. Публикация — одна транзакция.
При отдельном output сохраняется ПОЛНАЯ БД; копирование неизбежно и имеет
прогресс. Публикация отдельной БД — SQLite Backup API, не файловая подмена.

Нельзя параллельно изменять источник. В WAL долгое чтение может удерживать
WAL; в DELETE-журнале мешать writer. flock защищает только сотрудничающие
процессы. Изменения через другие соединения отслеживаются data_version;
подмена файла — по inode. Не удалять WAL/SHM вручную.

Настройки окружения:
 WAVE_ACTIVITY_DB              default big_scale/data/db/wave_activity.db
 WAVE_INDEX_OUTPUT_DB          default тот же путь
 WAVE_INDEX_OVERWRITE          false; только для отдельного output
 WAVE_INDEX_ALLOW_LEGACY       false
 WAVE_STORM_PERCENTILE         90
 WAVE_STORM_THRESHOLD          пусто = перцентиль
 WAVE_DT_SECONDS               86400
 WAVE_LOG_LEVEL                INFO
 WAVE_INDEX_FULL_CHECK         false; полный quick_check/foreign_key_check
 WAVE_INDEX_LOG_EVERY          250; прогресс чтения точек
 WAVE_INDEX_PROGRESS_SECONDS   5; прогресс backup/полного аудита

Нет полного COUNT(*) дневной таблицы, VACUUM и глобального fetchall().
Нули восстанавливаются базовым builder из n_days; исходные батиметрические
пропуски этим этапом не исправляются. Возможность возобновления метрик после
прерывания не реализована: повторный запуск заново считает метрики точек.
"""

import fcntl
import json
import math
import os
import sqlite3
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from loguru import logger


def find_project_root() -> Path:
    directory = Path(__file__).resolve().parent
    for candidate in (directory, *directory.parents):
        if (candidate / "src/waves/services/wave_exposure_db_service.py").is_file():
            return candidate
    raise RuntimeError("Не найден корень BreachTheBeach: src/waves/services/wave_exposure_db_service.py")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "big_scale/data"

INDEX_COLUMNS = (
    "point_id", "mean_cwef_wm", "e_storm_mjm", "storm_threshold_wm",
    "storm_percentile", "k_dir", "cv", "n_days", "n_storm_days",
    "top3_sectors", "r1", "r2", "r3", "r4", "wer",
)
METADATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS wave_exposure_build_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    created_at TEXT NOT NULL,
    config_json TEXT NOT NULL
)
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def env_path(name, default):
    raw = os.getenv(name, "").strip()
    path = (Path(raw) if raw else default).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def env_float(name, default):
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name}: ожидается число, получено {raw!r}") from error
    require(math.isfinite(value), f"{name} должен быть конечным.")
    return value


def env_optional_float(name):
    return env_float(name, 0.0) if os.getenv(name, "").strip() else None


def env_int(name, default):
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name}: ожидается целое число, получено {raw!r}") from error
    require(value > 0, f"{name} должен быть > 0.")
    return value


def env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}: ожидается true/false, получено {raw!r}")


def configure_logging():
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("WAVE_LOG_LEVEL", "INFO").strip().upper(),
               diagnose=False, backtrace=False,
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")


@contextmanager
def database_lock(path):
    lock_path = Path(f"{path}.lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"БД занята другим этапом: {path}; lock={lock_path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def connect_readonly(path):
    con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=60, isolation_level=None)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA cache_size=-65536")
    return con


def file_identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def data_version(con):
    return int(con.execute("PRAGMA data_version").fetchone()[0])


def assert_source_unchanged(watcher, version, source, identity):
    if file_identity(source) != identity or data_version(watcher) != version:
        raise RuntimeError("Исходная БД изменена или подменена во время расчёта. Индексы не опубликованы.")


def same_file(a, b):
    return a == b or (a.exists() and b.exists() and a.samefile(b))


def validate_paths(source, output):
    if source == output:
        return
    # Обратные пересечения служебных файлов тоже запрещены.
    source_files = [source] + [Path(f"{source}{s}") for s in ("-wal", "-shm", "-journal", ".lock")]
    output_files = [output] + [Path(f"{output}{s}") for s in ("-wal", "-shm", "-journal", ".lock")]
    for a in source_files:
        for b in output_files:
            require(not same_file(a, b), f"Пересечение входного/выходного/служебного файла: {a}, {b}")


def full_check(con, interval):
    logger.info("Полный аудит SQLite: может долго читать большую БД.")
    started = last = time.monotonic()
    def progress():
        nonlocal last
        now = time.monotonic()
        if now - last >= interval:
            logger.info("Аудит SQLite выполняется, {:.0f} с", now - started)
            last = now
        return 0
    con.set_progress_handler(progress, 100_000)
    try:
        for row in con.execute("PRAGMA quick_check"):
            require(row[0] == "ok", f"SQLite quick_check: {row[0]}")
        error = con.execute("PRAGMA foreign_key_check").fetchone()
        require(error is None, f"Нарушены внешние ключи: {error}")
    finally:
        con.set_progress_handler(None, 0)
    logger.info("Полный аудит завершён за {:.1f} с", time.monotonic() - started)


def validate_source(con, allow_legacy, audit, interval):
    from src.waves.storage.schema import SCHEMA_VERSION
    require(con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION, "Неподдерживаемая версия схемы wave_activity.")
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"wave_points", "wave_activity", "wave_activity_summary", "wave_exposure_index"}
    require(required <= tables, f"Нет таблиц: {sorted(required-tables)}")
    count = int(con.execute("SELECT COUNT(*) FROM wave_points").fetchone()[0])
    require(count > 0, "wave_points пуста.")
    missing = con.execute("""SELECT p.id FROM wave_points p
        LEFT JOIN wave_activity_summary s ON s.point_id=p.id
        WHERE s.point_id IS NULL LIMIT 20""").fetchall()
    require(not missing, f"Нет summary для точек: {missing}")
    orphan = con.execute("""SELECT s.point_id FROM wave_activity_summary s
        LEFT JOIN wave_points p ON p.id=s.point_id
        WHERE p.id IS NULL LIMIT 20""").fetchall()
    require(not orphan, f"Summary без точки: {orphan}")
    invalid = con.execute("""SELECT point_id FROM wave_activity_summary
        WHERE n_days IS NULL OR n_active_days IS NULL OR n_days<0
           OR n_active_days<0 OR n_active_days>n_days LIMIT 20""").fetchall()
    require(not invalid, f"Некорректные счётчики summary: {invalid}")
    checkpoints = {"wave_build_meta", "wave_build_progress"}
    present = checkpoints & tables
    require(not present or present == checkpoints, "Неполная структура checkpoint.")
    if present:
        row = con.execute("SELECT status FROM wave_build_meta WHERE id=1").fetchone()
        require(row is not None and row[0] == "completed", f"Расчёт wave_activity не завершён: {row}")
        bad = con.execute("""SELECT p.id FROM wave_points p
            LEFT JOIN wave_build_progress b ON b.point_id=p.id
            WHERE b.point_id IS NULL OR b.status IS NULL OR b.status<>'done' LIMIT 20""").fetchall()
        require(not bad, f"Незавершённые точки checkpoint: {bad}")
    else:
        require(allow_legacy, "Нет checkpoint. Для осознанной обработки старой БД: WAVE_INDEX_ALLOW_LEGACY=true.")
        logger.warning("Legacy-БД: summary проверена, независимого подтверждения завершённости нет.")
    plan = [str(r[3]) for r in con.execute("""EXPLAIN QUERY PLAN
        SELECT point_id,day,wind_azimuth_deg,cwef_wm FROM wave_activity
        WHERE point_id=? ORDER BY day""", (0,))]
    logger.info("План чтения одной точки: {}", plan)
    require(any("SEARCH" in item.upper() for item in plan)
            and not any("TEMP B-TREE" in item.upper() for item in plan),
            "Нет подходящего индексированного чтения по point_id/day. Исправь схему; полный scan на каждую точку запрещён.")
    if audit:
        full_check(con, interval)
    else:
        logger.info("Полный аудит отключён; для него WAVE_INDEX_FULL_CHECK=true.")
    logger.info("Источник проверен: точек={}; COUNT(*) дневной таблицы не выполняется.", count)
    logger.warning("WEI не восстанавливает исходные батиметрические пропуски, учтённые ранее как нули.")
    return count


class LazyActivityByPoint:
    """Совместимый .get() без загрузки дневных данных всех точек.

    В памяти остаётся один ряд точки. ORDER BY day сохраняет порядок
    исходного builder. Запрос использует PK (point_id,day).
    """
    def __init__(self, con, total, log_every):
        self.con, self.total, self.log_every = con, total, log_every
        self.processed = self.rows_read = 0
        self.started = time.monotonic()

    def get(self, point_id, default=None):
        from src.waves.storage import WaveActivityRow
        rows = [WaveActivityRow(point_id=int(r[0]), day=int(r[1]),
                                wind_azimuth_deg=int(r[2]), cwef_wm=int(r[3]))
                for r in self.con.execute("""SELECT point_id,day,wind_azimuth_deg,cwef_wm
                    FROM wave_activity WHERE point_id=? ORDER BY day""", (int(point_id),))]
        self.processed += 1
        self.rows_read += len(rows)
        if self.processed % self.log_every == 0 or self.processed == self.total:
            logger.info("Чтение дневных данных: {}/{} точек, {} строк, {:.1f} с",
                        self.processed, self.total, self.rows_read, time.monotonic()-self.started)
        return rows


class StreamingRepositoryAdapter:
    """Duck-typed repository для неизменённого WaveExposureDatabaseBuilder."""
    def __init__(self, source_con, result_repository, total, log_every):
        self._con = source_con
        self.result_repository = result_repository
        self.activity = LazyActivityByPoint(source_con, total, log_every)

    def assert_schema_version(self):
        from src.waves.storage import WaveActivityRepository
        return WaveActivityRepository.assert_schema_version(self)

    def read_summaries(self):
        from src.waves.storage import WaveActivityRepository
        return WaveActivityRepository.read_summaries(self)

    def read_activity_by_point(self):
        logger.info("Потоковый режим: дневные данные будут читаться по одной точке.")
        return self.activity

    def replace_exposure_indices(self, rows):
        return self.result_repository.replace_exposure_indices(rows)

    def commit(self):
        return self.result_repository.commit()

    def optimize(self):
        # Не запускать ANALYZE/optimize исходной 20+ ГБ БД.
        return None


def create_result_database(reader, destination):
    ddl_row = reader.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='wave_exposure_index'").fetchone()
    require(ddl_row and ddl_row[0], "Не найдена схема wave_exposure_index.")
    con = sqlite3.connect(destination)
    try:
        with con:
            con.execute("PRAGMA foreign_keys=ON")
            con.execute("CREATE TABLE wave_points(id INTEGER PRIMARY KEY)")
            con.executemany("INSERT INTO wave_points(id) VALUES (?)", reader.execute("SELECT id FROM wave_points ORDER BY id"))
            con.execute(ddl_row[0])
            con.execute(METADATA_TABLE_SQL)
    finally:
        con.close()


def validate_and_write_result(path, expected_points, stats, config):
    con = sqlite3.connect(path)
    try:
        require(stats.point_count == expected_points, "Число записанных индексов не совпадает с числом точек.")
        count = con.execute("SELECT COUNT(*) FROM wave_exposure_index").fetchone()[0]
        missing = con.execute("""SELECT p.id FROM wave_points p LEFT JOIN wave_exposure_index i
            ON i.point_id=p.id WHERE i.point_id IS NULL LIMIT 1""").fetchone()
        require(count == expected_points and missing is None, "Итоговая таблица индексов неполна.")
        require(con.execute("PRAGMA foreign_key_check").fetchone() is None, "Нарушены FK временной таблицы индексов.")
        for row in con.execute("PRAGMA quick_check"):
            require(row[0] == "ok", f"Повреждение временной БД: {row[0]}")
        for row in con.execute("SELECT * FROM wave_exposure_index"):
            require(all(not isinstance(v, float) or math.isfinite(v) for v in row), "Неконечное число в рассчитанных индексах.")
        with con:
            con.execute("INSERT OR REPLACE INTO wave_exposure_build_meta VALUES (1,?,?)",
                        (datetime.now(UTC).isoformat(), json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False)))
    finally:
        con.close()


def publish_indices(target, computed, check_unchanged=None):
    con = sqlite3.connect(target.as_uri()+"?mode=rw", uri=True, timeout=60, isolation_level=None)
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA synchronous=FULL")
        con.execute("ATTACH DATABASE ? AS computed", (computed.as_uri()+"?mode=ro",))
        con.execute("BEGIN IMMEDIATE")
        try:
            if check_unchanged is not None:
                check_unchanged()
            columns = ",".join(INDEX_COLUMNS)
            con.execute("DELETE FROM main.wave_exposure_index")
            con.execute(f"INSERT INTO main.wave_exposure_index ({columns}) SELECT {columns} FROM computed.wave_exposure_index")
            con.execute(METADATA_TABLE_SQL)
            con.execute("DELETE FROM main.wave_exposure_build_meta")
            con.execute("""INSERT INTO main.wave_exposure_build_meta(id,created_at,config_json)
                SELECT id,created_at,config_json FROM computed.wave_exposure_build_meta""")
            con.execute("COMMIT")
        except BaseException:
            if con.in_transaction:
                con.execute("ROLLBACK")
            raise
    finally:
        con.close()


def backup_with_progress(source_con, destination, interval, label):
    started = last = time.monotonic()
    page_size = source_con.execute("PRAGMA page_size").fetchone()[0]
    def progress(status, remaining, total):
        nonlocal last
        now = time.monotonic()
        if now-last < interval and remaining != 0:
            return
        last = now
        done = max(0, total-remaining)
        logger.info("{}: {:.1f}% | {:.2f}/{:.2f} GiB | {:.0f} с | status={}", label,
                    100*done/total if total else 0, done*page_size/1024**3,
                    total*page_size/1024**3, now-started, status)
    target = sqlite3.connect(destination, timeout=60)
    try:
        target.execute("PRAGMA synchronous=FULL")
        source_con.backup(target, pages=4096, progress=progress, sleep=0.1)
    finally:
        target.close()


def main():
    configure_logging()
    source = env_path("WAVE_ACTIVITY_DB", DATA_DIR / "db/wave_activity.db")
    output = env_path("WAVE_INDEX_OUTPUT_DB", source)
    percentile = env_float("WAVE_STORM_PERCENTILE", 90.0)
    threshold = env_optional_float("WAVE_STORM_THRESHOLD")
    dt = env_float("WAVE_DT_SECONDS", 86_400.0)
    overwrite = env_bool("WAVE_INDEX_OVERWRITE")
    legacy = env_bool("WAVE_INDEX_ALLOW_LEGACY")
    audit = env_bool("WAVE_INDEX_FULL_CHECK")
    every = env_int("WAVE_INDEX_LOG_EVERY", 250)
    interval = env_float("WAVE_INDEX_PROGRESS_SECONDS", 5.0)
    require(0 < percentile < 100, "WAVE_STORM_PERCENTILE должен быть в (0,100).")
    require(threshold is None or threshold >= 0, "WAVE_STORM_THRESHOLD должен быть >= 0.")
    require(dt > 0 and interval > 0, "DT и интервал прогресса должны быть > 0.")
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_paths(source, output)
    in_place = source == output
    logger.info("Вход: {}", source)
    logger.info("Выход: {}", output)
    logger.info("Режим: {}", "обновление индексов БЕЗ полной копии БД" if in_place else "полная отдельная БД; потребуются два прохода backup")
    logger.info("Шторм: percentile={}, threshold={}, dt={} с", percentile, threshold, dt)
    from src.waves.services.wave_exposure_db_service import WaveExposureDatabaseBuilder
    from src.waves.storage import WaveActivityRepository
    started = time.monotonic()
    with ExitStack() as stack:
        for path in sorted({source, output}, key=str):
            stack.enter_context(database_lock(path))
        validate_paths(source, output)
        if not in_place and output.exists() and not overwrite:
            raise FileExistsError(f"Выход существует: {output}. Для замены WAVE_INDEX_OVERWRITE=true.")
        identity = file_identity(source)
        watcher = connect_readonly(source)
        stack.callback(watcher.close)
        version = data_version(watcher)
        reader = connect_readonly(source)
        stack.callback(reader.close)
        unchanged = lambda: assert_source_unchanged(watcher, version, source, identity)
        # Первый SELECT фиксирует snapshot; watcher остаётся отдельным соединением
        # без длительной транзакции, чтобы data_version мог обнаружить изменения.
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        unchanged()
        logger.info("Согласованный read-snapshot открыт; journal_mode={}", reader.execute("PRAGMA journal_mode").fetchone()[0])
        with TemporaryDirectory(prefix=f".{output.name}.wei-stream-", dir=output.parent) as directory:
            computed = Path(directory) / "indices.db"
            expected = validate_source(reader, legacy, audit, interval)
            unchanged()
            create_result_database(reader, computed)
            logger.info("Расчёт метрик: дневная таблица не копируется и не загружается целиком в RAM.")
            with WaveActivityRepository(computed) as result_repository:
                adapter = StreamingRepositoryAdapter(reader, result_repository, expected, every)
                stats = WaveExposureDatabaseBuilder(repository=adapter, storm_percentile=percentile,
                                                    storm_threshold=threshold, dt_seconds=dt).run()
            unchanged()
            validate_and_write_result(computed, expected, stats, {
                "source": str(source), "builder": "WaveExposureDatabaseBuilder",
                "io_mode": "per_point_read_snapshot_v1", "storm_percentile": percentile,
                "storm_threshold_wm": threshold, "dt_seconds": dt,
                "point_count": stats.point_count, "valid_index_count": stats.valid_index_count,
                "stored_activity_rows_read": adapter.activity.rows_read,
                "full_integrity_check": audit,
                "activity_representation": "stored_positive_integer_CWEF_plus_reconstructed_zeros",
                "missing_bathymetry_repaired": False,
            })
            if in_place:
                # Закрыть read-транзакцию ДО начала writer (важно для DELETE journal).
                reader.execute("ROLLBACK")
                logger.info("Публикация только индексов и метаданных одной транзакцией.")
                publish_indices(source, computed, unchanged)
            else:
                full = Path(directory) / "full-output.db"
                logger.warning("Отдельный output — полная копия. Нужны место под временную полную БД и место для output.")
                # Backup использует ту же согласованную read-транзакцию.
                backup_with_progress(reader, full, interval, "Снимок полной БД")
                reader.execute("ROLLBACK")
                unchanged()
                publish_indices(full, computed)
                unchanged()
                full_reader = connect_readonly(full)
                try:
                    backup_with_progress(full_reader, output, interval, "Публикация полной БД")
                finally:
                    full_reader.close()
    logger.success("Готово: points={}, valid_WER={}, min_WER={}, max_WER={}, mean_WER={}, {:.1f} с, db={}",
                   stats.point_count, stats.valid_index_count, stats.min_wer, stats.max_wer,
                   stats.mean_wer, time.monotonic()-started, output)
    if stats.valid_index_count != stats.point_count:
        logger.warning("У {} точек WER не определён; проверь компоненты и ранжирование.", stats.point_count-stats.valid_index_count)


if __name__ == "__main__":
    main()
