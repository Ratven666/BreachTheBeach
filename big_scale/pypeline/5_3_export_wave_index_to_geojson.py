from __future__ import annotations

"""Экспорт WEI/WER в GeoJSON с использованием базового экспортера.

Не экспортирует периоды из wave_activity_indices.db.
Источник должен содержать wave_points и wave_exposure_index.

Переменные окружения:
    WAVE_INDEX_DB          — приоритетный путь к БД с WEI/WER;
    WAVE_ACTIVITY_DB       — совместимый прежний параметр пути;
    WAVE_INDEX_GEOJSON     — выходной GeoJSON;
    WAVE_EXPORT_MODE       — points | summary;
    WAVE_GEOJSON_INDENT    — пусто: компактно; иначе 0..8;
    WAVE_GEOJSON_OVERWRITE — разрешить замену существующего GeoJSON;
    WAVE_EXPORT_ALLOW_LEGACY — разрешить БД без checkpoint;
    WAVE_LOG_LEVEL.

SQLite открывается только для чтения. Все SELECT выполняются
в одной read-транзакции. Публикуется только проверенный GeoJSON.
Блокировки fcntl совместимы с предыдущими этапами (macOS/Linux).
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
        if (candidate / "src/waves/export/geojson_exporter.py").is_file():
            return candidate
    raise RuntimeError("Не найден корень проекта BreachTheBeach.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "big_scale" / "data"


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}: ожидается true/false, получено {raw!r}.")


def parse_indent() -> int | None:
    raw = os.getenv("WAVE_GEOJSON_INDENT", "").strip()
    if not raw:
        return None
    value = int(raw)
    if not 0 <= value <= 8:
        raise ValueError("WAVE_GEOJSON_INDENT должен находиться в 0..8.")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@contextmanager
def file_lock(path: Path):
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Файл занят другим этапом: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def assert_paths_safe(source: Path, output: Path) -> None:
    require(output.suffix.lower() in {".geojson", ".json"}, "Выход должен иметь расширение .geojson или .json.")
    candidates = [output, Path(f"{output}.lock")]
    protected = [source] + [Path(f"{source}{suffix}") for suffix in ("-wal", "-shm", "-journal", ".lock")]
    for candidate in candidates:
        for item in protected:
            same = candidate == item
            if candidate.exists() and item.exists():
                same = same or candidate.samefile(item)
            require(not same, f"Выход или его lock совпадает с входным файлом: {candidate}")


def validate_database(connection, mode: str, allow_legacy: bool):
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    needed = {"wave_points", "wave_exposure_index"}
    if mode == "summary":
        needed.add("wave_activity_summary")
    require(
        needed <= tables,
        f"Нет таблиц {sorted(needed - tables)}. Нужна БД с WEI/WER, а не wave_activity_indices.db.",
    )
    count = int(connection.execute("SELECT COUNT(*) FROM wave_points").fetchone()[0])
    require(count > 0, "wave_points пуста.")
    missing = connection.execute("""
        SELECT p.id FROM wave_points p
        LEFT JOIN wave_exposure_index i ON i.point_id=p.id
        WHERE i.point_id IS NULL LIMIT 20
    """).fetchall()
    require(not missing, f"Нет WEI/WER для точек: {missing}. Сначала заверши расчёт wave_exposure_index.")
    orphan = connection.execute("""
        SELECT i.point_id FROM wave_exposure_index i
        LEFT JOIN wave_points p ON p.id=i.point_id
        WHERE p.id IS NULL LIMIT 20
    """).fetchall()
    require(not orphan, f"Индексы ссылаются на неизвестные точки: {orphan}")
    if mode == "summary":
        missing = connection.execute("""
            SELECT p.id FROM wave_points p
            LEFT JOIN wave_activity_summary s ON s.point_id=p.id
            WHERE s.point_id IS NULL LIMIT 20
        """).fetchall()
        require(not missing, f"Нет summary для точек: {missing}")
    checkpoint = {"wave_build_meta", "wave_build_progress"}
    present = checkpoint & tables
    require(not present or present == checkpoint, "Неполная структура checkpoint.")
    quality = {}
    if present:
        meta = connection.execute("SELECT status FROM wave_build_meta WHERE id=1").fetchone()
        require(meta is not None and meta[0] == "completed", "Расчёт волн не завершён.")
        missing = connection.execute("""
            SELECT p.id FROM wave_points p
            LEFT JOIN wave_build_progress b ON b.point_id=p.id
            WHERE b.point_id IS NULL OR b.status<>'done' LIMIT 20
        """).fetchall()
        require(not missing, f"Есть незавершённые точки: {missing}")
        quality = {
            int(pid): {"missing_bathymetry_days": int(no_bathy), "skipped_weather_rows": int(skipped)}
            for pid, no_bathy, skipped in connection.execute(
                "SELECT point_id,no_bathy,skipped FROM wave_build_progress"
            )
        }
    else:
        require(allow_legacy, "Нет checkpoint. Для старой БД явно задай WAVE_EXPORT_ALLOW_LEGACY=true.")
        logger.warning("Legacy-БД: завершённость расчёта независимо не подтверждается.")
    return count, quality


def reject_constant(value):
    raise ValueError(f"В GeoJSON обнаружено недопустимое число {value}.")


def validate_json_numbers(value, location="root") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"Неконечное число: {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            validate_json_numbers(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_numbers(item, f"{location}[{index}]")


def finalize_geojson(path, expected_count, source, mode, quality, indent):
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    require(data.get("type") == "FeatureCollection", "Ожидался FeatureCollection.")
    features = data.get("features")
    require(isinstance(features, list) and len(features) == expected_count, "Не совпадает число экспортированных точек.")
    seen = set()
    for feature in features:
        require(feature.get("type") == "Feature", "Некорректный Feature.")
        geometry = feature.get("geometry", {})
        require(geometry.get("type") == "Point", "Ожидалась геометрия Point.")
        coordinates = geometry.get("coordinates")
        require(isinstance(coordinates, list) and len(coordinates) == 2, "Некорректные координаты.")
        lon, lat = coordinates
        require(
            all(type(value) in (int, float) and math.isfinite(value) for value in coordinates)
            and -180 <= lon <= 180 and -90 <= lat <= 90,
            f"Координаты вне WGS84: {coordinates}",
        )
        properties = feature["properties"]
        pid = properties["point_id"]
        require(type(pid) is int and pid not in seen, f"Повторный или некорректный point_id={pid}")
        seen.add(pid)
        feature["id"] = pid
        if quality:
            require(pid in quality, f"Нет checkpoint для point_id={pid}")
            properties.update(quality[pid])
        if mode == "summary":
            # Прежнее имя mean_cwef_wm_s сохраняется для совместимости.
            properties["mean_cwef_active_days_wm"] = properties.get("mean_cwef_wm_s")
    data.pop("crs", None)
    missing_bathy = sum(item["missing_bathymetry_days"] for item in quality.values()) if quality else None
    data["metadata"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_db": str(source),
        "export_mode": mode,
        "coordinate_order": "longitude, latitude",
        "source_missing_bathymetry_point_days": missing_bathy,
        "source_quality": (
            "unknown_without_checkpoint" if missing_bathy is None else
            "bathymetry_gaps_present" if missing_bathy else "no_reported_bathymetry_gaps"
        ),
        "mean_cwef_wm_definition": "WEI mean of stored daily integer CWEF with reconstructed zero days",
        "mean_cwef_wm_s_definition": "summary mean of original positive CWEF days only",
        "mean_cwef_active_days_wm_alias": "mean_cwef_wm_s",
    }
    validate_json_numbers(data)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=indent,
                  separators=(",", ":") if indent is None else None)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return missing_bathy


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("WAVE_LOG_LEVEL", "INFO").strip().upper(),
               diagnose=False, backtrace=False)
    fallback = env_path("WAVE_ACTIVITY_DB", DATA_DIR / "db/wave_activity.db")
    source = env_path("WAVE_INDEX_DB", fallback)
    output = env_path("WAVE_INDEX_GEOJSON", DATA_DIR / "wave/wave_exposure_index.geojson")
    mode = os.getenv("WAVE_EXPORT_MODE", "summary").strip().lower()
    indent = parse_indent()
    overwrite = env_bool("WAVE_GEOJSON_OVERWRITE", False)
    legacy = env_bool("WAVE_EXPORT_ALLOW_LEGACY", False)
    require(mode in {"points", "summary"}, "WAVE_EXPORT_MODE: points | summary.")
    if not source.is_file():
        raise FileNotFoundError(f"БД с рассчитанными WEI/WER не найдена: {source}")
    assert_paths_safe(source, output)
    output.parent.mkdir(parents=True, exist_ok=True)

    from src.waves.export import WaveIndexGeoJSONExporter
    from src.waves.storage import WaveActivityRepository

    class ReadOnlyWaveRepository(WaveActivityRepository):
        # Базовые read_* и проверка схемы остаются без изменения.
        def __init__(self, path):
            self.path = Path(path).resolve()
            self._con = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=60)
            try:
                self._con.execute("PRAGMA query_only=ON")
                self._con.execute("BEGIN")
            except BaseException:
                self._con.close()
                raise

    logger.info("БД с WEI/WER: {}", source)
    logger.info("GeoJSON: {}; mode={}; indent={}", output, mode, indent)
    with ExitStack() as stack:
        locks = {Path(f"{source}.lock"), Path(f"{output}.lock")}
        for path in sorted(locks, key=str):
            stack.enter_context(file_lock(path))
        if output.exists() and not overwrite:
            raise FileExistsError(f"GeoJSON уже существует: {output}. Задай WAVE_GEOJSON_OVERWRITE=true.")
        with TemporaryDirectory(prefix=f".{output.stem}.export-", dir=output.parent) as directory:
            temporary = Path(directory) / "validated.geojson"
            with ReadOnlyWaveRepository(source) as repository:
                repository.assert_schema_version()
                count, quality = validate_database(repository._con, mode, legacy)
                stats = WaveIndexGeoJSONExporter(repository).export(
                    output_path=temporary, mode=mode, indent=indent,
                )
                missing_bathy = finalize_geojson(temporary, count, source, mode, quality, indent)
            # Оба файла находятся на одной файловой системе.
            # Прежний GeoJSON сохраняется до успешной валидации нового.
            os.replace(temporary, output)
    logger.success("Готово: features={}, файл={}", stats.feature_count, output)
    logger.info(
        "WER: min={}, max={}, mean={}",
        "не определён" if stats.min_wer is None else f"{stats.min_wer:.3f}",
        "не определён" if stats.max_wer is None else f"{stats.max_wer:.3f}",
        "не определён" if stats.mean_wer is None else f"{stats.mean_wer:.3f}",
    )
    if missing_bathy:
        logger.warning(
            "В источнике {} точко-дней с отсутствующей батиметрией. "
            "Ограничение отражено в metadata и свойствах точек; экспорт его не исправляет.",
            missing_bathy,
        )


if __name__ == "__main__":
    main()
