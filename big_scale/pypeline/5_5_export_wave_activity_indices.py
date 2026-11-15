from __future__ import annotations

"""Шаг 5.5: экспорт периодов и трендов волновой активности в GeoJSON.

Читает big_scale/data/db/wave_activity_indices.db из шага 5.4.
Использует export_geojson базового проекта без изменения src/.
Настройки задаются константами ниже.

Режимы:
    trends  — одна Feature на точку;
    periods — одна Feature на пару (точка, период).

Создаёт согласованный снимок SQLite, проверяет схему и полноту,
экспортирует во временный файл и только затем заменяет результат.
Входная БД открывается только для чтения; WAL учитывается SQLite.
Ни WAL, ни SHM вручную не удаляются.

macOS/Linux: используются согласованные блокировки <path>.lock.
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


def find_project_root() -> Path:
    directory = Path(__file__).resolve().parent
    for candidate in (directory, *directory.parents):
        if (candidate / "src/waves/activity_indices/exporter.py").is_file():
            return candidate
    raise RuntimeError("Не найден корень BreachTheBeach.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==================== НАСТРОЙКИ ====================

DATA_DIR = PROJECT_ROOT / "big_scale" / "data"
INDICES_DB = DATA_DIR / "db/wave_activity_indices.db"

EXPORT_MODE = "trends"  # trends | periods

# None: имя автоматически соответствует EXPORT_MODE.
# Можно явно задать Path("big_scale/data/wave/custom.geojson").
OUTPUT_GEOJSON: Path | None = None

GEOJSON_INDENT: int | None = None
OVERWRITE_OUTPUT = False
LOG_LEVEL = "INFO"


# ==================== ОБЩИЕ ФУНКЦИИ ====================

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_settings() -> None:
    require(EXPORT_MODE in {"trends", "periods"}, "EXPORT_MODE: trends | periods.")
    require(isinstance(OVERWRITE_OUTPUT, bool), "OVERWRITE_OUTPUT должен быть bool.")
    require(
        GEOJSON_INDENT is None or (
            type(GEOJSON_INDENT) is int and 0 <= GEOJSON_INDENT <= 8
        ),
        "GEOJSON_INDENT должен быть None или целым числом в 0..8.",
    )


def assert_paths_safe(source: Path, output: Path) -> None:
    require(output.suffix.lower() in {".geojson", ".json"}, "Выход должен иметь расширение .geojson или .json.")
    protected = [source] + [Path(f"{source}{s}") for s in ("-wal", "-shm", "-journal", ".lock")]
    for candidate in (output, Path(f"{output}.lock")):
        for path in protected:
            same = candidate == path
            if candidate.exists() and path.exists():
                same = same or candidate.samefile(path)
            require(not same, f"Выход или его lock совпадает с входным файлом: {candidate}")


@contextmanager
def file_lock(path: Path):
    # Lock-файл не удаляется: блокировку снимает ОС.
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Файл занят другим этапом пайплайна: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=60)


def snapshot_database(source: Path, snapshot: Path) -> None:
    reader = connect_readonly(source)
    writer = None
    try:
        writer = sqlite3.connect(snapshot, timeout=60)
        reader.backup(writer, pages=4096, sleep=0.1)
    finally:
        if writer is not None:
            writer.close()
        reader.close()


def validate_database(path: Path, expected_version: int, epoch: str):
    connection = connect_readonly(path)
    try:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        require(version == expected_version, f"Версия индексов {version}; ожидается {expected_version}.")
        for row in connection.execute("PRAGMA quick_check"):
            require(row[0] == "ok", f"SQLite quick_check: {row[0]}")
        violation = connection.execute("PRAGMA foreign_key_check").fetchone()
        require(violation is None, f"Нарушены внешние ключи: {violation}")
        tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        needed = {"meta", "points", "periods", "point_period_indices", "point_trends"}
        require(
            needed <= tables,
            f"Нет таблиц {sorted(needed - tables)}. Нужна wave_activity_indices.db из шага 5.4, не БД WEI/WER.",
        )
        metadata = dict(connection.execute("SELECT key,value FROM meta"))
        require(metadata.get("epoch") == epoch, "Эпоха дат в БД не соответствует модулю проекта.")
        points = int(connection.execute("SELECT COUNT(*) FROM points").fetchone()[0])
        periods = int(connection.execute("SELECT COUNT(*) FROM periods").fetchone()[0])
        require(points > 0 and periods > 0, "В БД нет точек или периодов.")
        trends = int(connection.execute("SELECT COUNT(*) FROM point_trends").fetchone()[0])
        rows = int(connection.execute("SELECT COUNT(*) FROM point_period_indices").fetchone()[0])
        require(trends == points, f"Таблица трендов неполна: {trends} вместо {points}.")
        require(rows == points * periods, f"Таблица периодных индексов неполна: {rows} вместо {points * periods}.")
        bad = connection.execute("""
            SELECT i.point_id,i.period_id FROM point_period_indices i
            JOIN periods p USING(period_id)
            WHERE i.n_active_days>p.n_days LIMIT 20
        """).fetchall()
        require(not bad, f"Активных дней больше длины периода: {bad}")
        logger.info("Снимок проверен: точек={}, периодов={}, периодных записей={}", points, periods, rows)
        return points, periods, metadata
    finally:
        connection.close()


def reject_constant(value):
    raise ValueError(f"Недопустимое JSON-число: {value}.")


def validate_numbers(value, location="root") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"Неконечное число: {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            validate_numbers(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_numbers(item, f"{location}[{index}]")


def finalize_geojson(path: Path, expected: int, source: Path, metadata: dict):
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    require(data.get("type") == "FeatureCollection", "Ожидался FeatureCollection.")
    features = data.get("features")
    require(isinstance(features, list) and len(features) == expected, "Неверное количество Feature.")
    seen = set()
    for feature in features:
        require(feature.get("type") == "Feature", "Некорректный Feature.")
        geometry = feature.get("geometry", {})
        require(geometry.get("type") == "Point", "Ожидалась геометрия Point.")
        coordinates = geometry.get("coordinates")
        require(isinstance(coordinates, list) and len(coordinates) == 2, "Некорректные координаты.")
        lon, lat = coordinates
        require(
            all(type(v) in (int, float) and math.isfinite(v) for v in coordinates)
            and -180 <= lon <= 180 and -90 <= lat <= 90,
            f"Некорректные координаты WGS84: {coordinates}",
        )
        properties = feature.get("properties", {})
        point_id = properties.get("point_id")
        require(type(point_id) is int, "Некорректный point_id.")
        if EXPORT_MODE == "periods":
            period_id = properties.get("period_id")
            require(type(period_id) is int, "Некорректный period_id.")
            key = (point_id, period_id)
            feature["id"] = f"{point_id}:{period_id}"
        else:
            key = point_id
            feature["id"] = point_id
        require(key not in seen, f"Повторная Feature: {key}")
        seen.add(key)
    data.pop("crs", None)
    data["metadata"] = {
        "export_created_at": datetime.now(UTC).isoformat(),
        "source_indices_db": str(source),
        "export_mode": EXPORT_MODE,
        "coordinate_order": "longitude, latitude",
        "indices_metadata": metadata,
    }
    validate_numbers(data)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False,
                  indent=GEOJSON_INDENT,
                  separators=(",", ":") if GEOJSON_INDENT is None else None)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL, diagnose=False, backtrace=False)
    validate_settings()
    source = resolve_path(INDICES_DB)
    output = resolve_path(
        OUTPUT_GEOJSON if OUTPUT_GEOJSON is not None else
        DATA_DIR / "wave" / f"wave_activity_indices_{EXPORT_MODE}.geojson"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Нет БД индексов: {source}. Сначала выполни шаг 5.4.")
    assert_paths_safe(source, output)
    output.parent.mkdir(parents=True, exist_ok=True)

    from src.waves.activity_indices import export_geojson
    from src.waves.activity_indices.constants import EPOCH, SCHEMA_VERSION

    logger.info("Вход: {}", source)
    logger.info("Выход: {}; режим={}", output, EXPORT_MODE)
    with ExitStack() as stack:
        for lock in sorted({Path(f"{source}.lock"), Path(f"{output}.lock")}, key=str):
            stack.enter_context(file_lock(lock))
        if output.exists() and not OVERWRITE_OUTPUT:
            raise FileExistsError(f"Файл уже существует: {output}. Для замены задай OVERWRITE_OUTPUT=True.")
        with TemporaryDirectory(prefix=f".{output.stem}.export-", dir=output.parent) as directory:
            directory = Path(directory)
            snapshot = directory / "indices_snapshot.db"
            temporary = directory / "validated.geojson"
            snapshot_database(source, snapshot)
            points, periods, metadata = validate_database(snapshot, SCHEMA_VERSION, EPOCH.isoformat())
            expected = points if EXPORT_MODE == "trends" else points * periods
            gaps = metadata.get("source_missing_bathymetry_point_days")
            if gaps is not None:
                gap_count = int(gaps)
                require(gap_count >= 0, "Некорректный счётчик батиметрических пропусков в meta.")
                if gap_count:
                    logger.warning("В исходном расчёте {} точко-дней с отсутствующей батиметрией. Метаданные сохранены в GeoJSON.", gap_count)
            else:
                logger.warning("В meta нет счётчика батиметрических пропусков: качество источника по этому признаку неизвестно.")
            stats = export_geojson(snapshot, temporary, mode=EXPORT_MODE, indent=GEOJSON_INDENT)
            require(stats.feature_count == expected, "Базовый экспортёр вернул неполное число объектов.")
            finalize_geojson(temporary, expected, source, metadata)
            os.replace(temporary, output)
    logger.success("Готово: {} features -> {}", stats.feature_count, output)


if __name__ == "__main__":
    main()
