from __future__ import annotations

"""Local Moran для протяжённого берега: геодезические расстояния WGS84.

Полная замена big_scale/pypeline/5_6_local_moran.py.
Не требует одной UTM и не использует EPSG:3857 для расстояний.
Формулы Local Moran, перестановки и BH-FDR остаются в базовом проекте.
Локально на время вызова сервиса подменяется только функция подготовки
геометрии. Файлы src/ не изменяются.

Каждая часть исходной линии остаётся отдельным сегментом соседства.
Узлы полилинии соединяются кратчайшими геодезическими дугами WGS84.
Склейка разорванных частей и построение графа береговой сети не выполняются.

Результаты сначала рассчитываются на SQLite-копии и проверяются.
Внешние приложения, записывающие в INDEX_DB, нужно закрыть.
Блокировки .lock согласованы с предыдущими этапами (macOS/Linux).
"""

import fcntl
import hashlib
import importlib
import json
import math
import os
import sqlite3
import sys
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from loguru import logger


def find_project_root():
    directory = Path(__file__).resolve().parent
    for candidate in (directory, *directory.parents):
        if (candidate / "src/waves/local_moran/service.py").is_file():
            return candidate
    raise RuntimeError("Не найден корень BreachTheBeach.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "big_scale" / "data"
MODE = "calculate_export"  # calculate | export | calculate_export
INDEX_DB = DATA_DIR / "db/wave_activity_indices.db"

# Укажи тот же береговой файл, по которому были созданы текущие точки.
# None: единственный *.geojson из big_scale/data/coastline.
COASTLINE_GEOJSON: Path | None = DATA_DIR / "S2CoastRussiaCropped.geojson"
OUTPUT_GEOJSON = DATA_DIR / "wave/local_moran.geojson"

RADIUS_M = 100000.0
WINDOW_WIDTH_M = 200000.0
ALONGSHORE_MODE = "abs"
MAX_SNAP_DISTANCE_M: float | None = 100.0
STRICT_SNAP = False
WEIGHT_MODE = "binary"
PERMUTATIONS = 999
RANDOM_SEED = 42
ALPHA_RAW = 0.999
ALPHA_FDR = 0.999
EXPORT_RUN_ID: int | None = None
OVERWRITE_GEOJSON = False
GEOJSON_INDENT: int | None = None

# Техническое разбиение длинных рёбер для пространственного индекса.
# Геометрическая связность исходной линии при этом НЕ меняется.
MAX_INDEX_EDGE_M = 10_000.0
MAX_INDEX_PIECES = 5_000_000
SNAP_OPTIMIZATION_TOLERANCE_M = 0.01
GEODESIC_METHOD = "WGS84_GEODESIC_CHAINAGE_V1"
LOG_LEVEL = "INFO"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def validate_settings():
    require(MODE in {"calculate", "export", "calculate_export"}, "Некорректный MODE.")
    require(ALONGSHORE_MODE in {"abs", "net"}, "ALONGSHORE_MODE: abs | net.")
    require(WEIGHT_MODE in {"binary", "inverse_distance"}, "Некорректный WEIGHT_MODE.")
    for name, value in (("RADIUS_M", RADIUS_M), ("WINDOW_WIDTH_M", WINDOW_WIDTH_M),
                        ("MAX_INDEX_EDGE_M", MAX_INDEX_EDGE_M),
                        ("SNAP_OPTIMIZATION_TOLERANCE_M", SNAP_OPTIMIZATION_TOLERANCE_M)):
        require(math.isfinite(value) and value > 0, f"{name} должен быть конечным и > 0.")
    require(MAX_INDEX_EDGE_M <= 100_000, "MAX_INDEX_EDGE_M не должен превышать 100000 м.")
    require(WINDOW_WIDTH_M >= 2 * RADIUS_M, "WINDOW_WIDTH_M должен быть >= 2 * RADIUS_M.")
    if MAX_SNAP_DISTANCE_M is not None:
        require(math.isfinite(MAX_SNAP_DISTANCE_M) and MAX_SNAP_DISTANCE_M > 0, "Некорректный MAX_SNAP_DISTANCE_M.")
    for value in (ALPHA_RAW, ALPHA_FDR):
        require(math.isfinite(value) and 0 < value < 1, "Пороги alpha должны быть в (0, 1).")
        if value > 0.1:
            logger.warning("Высокий порог значимости {}. Для просмотра всех квадрантов используй quadrant.", value)
    require(type(PERMUTATIONS) is int and PERMUTATIONS >= 99, "PERMUTATIONS должен быть >= 99.")
    require(type(RANDOM_SEED) is int and RANDOM_SEED >= 0, "Некорректный RANDOM_SEED.")
    require(type(MAX_INDEX_PIECES) is int and MAX_INDEX_PIECES > 0, "Некорректный MAX_INDEX_PIECES.")
    require(type(STRICT_SNAP) is bool and type(OVERWRITE_GEOJSON) is bool, "Некорректные bool-настройки.")
    require(EXPORT_RUN_ID is None or (type(EXPORT_RUN_ID) is int and EXPORT_RUN_ID > 0), "Некорректный EXPORT_RUN_ID.")
    require(GEOJSON_INDENT is None or (type(GEOJSON_INDENT) is int and 0 <= GEOJSON_INDENT <= 8), "Некорректный GEOJSON_INDENT.")


@contextmanager
def file_lock(path):
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Файл занят другим этапом: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def readonly(path):
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=60)


def copy_database(source, destination):
    reader, writer = readonly(source), None
    try:
        writer = sqlite3.connect(destination, timeout=60)
        writer.execute("PRAGMA synchronous=FULL")
        reader.backup(writer, pages=4096, sleep=0.1)
    finally:
        if writer is not None:
            writer.close()
        reader.close()


def file_state(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def resolve_coastline():
    if COASTLINE_GEOJSON is None:
        files = sorted((DATA_DIR / "coastline").glob("*.geojson"))
        require(len(files) == 1, "Задай COASTLINE_GEOJSON явно: в data/coastline нет единственного GeoJSON.")
        path = files[0].resolve()
    else:
        path = resolve_path(COASTLINE_GEOJSON)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def iter_line_parts(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield geometry
    elif geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from iter_line_parts(part)
    else:
        raise ValueError(f"Берег должен содержать линии, получено {geometry.geom_type}.")


class GeodesicCoastIndex:
    """Индекс геодезических рёбер, без плоской проекции территории.

    ECEF используется только для отбора кандидатов по хордовым нижним
    оценкам. Итоговые snap-расстояния и chainage вычисляются Geod.
    Привязка численная: точность зависит от допуска оптимизации.
    """
    def __init__(self, parts):
        import numpy as np
        from pyproj import Geod, Transformer
        from scipy.spatial import cKDTree
        self.np = np
        self.geod = Geod(ellps="WGS84")
        self.ecef = Transformer.from_crs(4979, 4978, always_xy=True)
        self.segments = []
        arrays = []
        total = 0
        for segment_id, coordinates in parts:
            xy = np.asarray(coordinates, dtype=float)[:, :2]
            require(len(xy) >= 2 and np.isfinite(xy).all(), "Некорректные вершины линии.")
            require((np.abs(xy[:, 0]) <= 180).all() and (np.abs(xy[:, 1]) <= 90).all(), "Вершины вне WGS84.")
            azimuth, _, lengths = self.geod.inv(xy[:-1, 0], xy[:-1, 1], xy[1:, 0], xy[1:, 1])
            lengths = np.asarray(lengths)
            require(np.isfinite(lengths).all(), "Не удалось вычислить длины рёбер.")
            starts = np.r_[0.0, np.cumsum(lengths)[:-1]]
            valid = lengths > 1e-6
            if not valid.any():
                continue
            points, azimuth, starts, lengths = xy[:-1][valid], np.asarray(azimuth)[valid], starts[valid], lengths[valid]
            counts = np.ceil(lengths / MAX_INDEX_EDGE_M).astype(np.int64)
            count = int(counts.sum())
            total += count
            require(total <= MAX_INDEX_PIECES, "Превышен MAX_INDEX_PIECES; индекс слишком велик.")
            edge = np.repeat(np.arange(len(lengths)), counts)
            offsets = np.arange(count) - np.repeat(np.cumsum(counts) - counts, counts)
            piece_lengths = lengths[edge] / counts[edge]
            distances = offsets * piece_lengths
            lon, lat, _ = self.geod.fwd(points[edge, 0], points[edge, 1], azimuth[edge], distances)
            end_lon, end_lat, _ = self.geod.fwd(points[edge, 0], points[edge, 1], azimuth[edge], distances + piece_lengths)
            local_azimuth, _, _ = self.geod.inv(lon, lat, end_lon, end_lat)
            mid_lon, mid_lat, _ = self.geod.fwd(lon, lat, local_azimuth, piece_lengths / 2)
            rank = len(self.segments)
            self.segments.append({"id": str(segment_id), "length": float(lengths.sum()),
                                  "closed": bool(np.array_equal(xy[0], xy[-1]))})
            arrays.append(np.column_stack((lon, lat, local_azimuth, piece_lengths,
                                           starts[edge] + distances, np.full(count, rank), mid_lon, mid_lat)))
        require(bool(arrays), "В береговом файле нет ненулевых линий.")
        self.edges = np.concatenate(arrays)
        self.midpoints = np.column_stack(self.ecef.transform(
            self.edges[:, 6], self.edges[:, 7], np.zeros(len(self.edges))))
        self.tree = cKDTree(self.midpoints)
        self.max_half_length = float(self.edges[:, 3].max() / 2)
        logger.info("Геодезический индекс: частей={}, рёбер={}", len(self.segments), len(self.edges))

    def closest_on_edge(self, edge_index, lon, lat):
        from scipy.optimize import minimize_scalar
        row = self.edges[edge_index]
        def distance(s):
            x, y, _ = self.geod.fwd(float(row[0]), float(row[1]), float(row[2]), float(s))
            return float(self.geod.inv(lon, lat, x, y)[2])
        length = float(row[3])
        result = minimize_scalar(distance, bounds=(0.0, length), method="bounded",
                                 options={"xatol": SNAP_OPTIMIZATION_TOLERANCE_M})
        require(bool(result.success), "Не сошлась геодезическая привязка к ребру.")
        choices = [(distance(0.0), 0.0), (distance(length), length), (float(result.fun), float(result.x))]
        value, along = min(choices)
        return value, along

    def nearest(self, lon, lat):
        np = self.np
        require(math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90,
                "Некорректная координата береговой точки.")
        xyz = np.asarray(self.ecef.transform(lon, lat, 0.0))
        _, seed = self.tree.query(xyz)
        seed = int(seed)
        best_distance, best_s = self.closest_on_edge(seed, lon, lat)
        best_edge = seed
        # Хорда не длиннее поверхностного пути. Это не выбор фиксированных k соседей.
        candidates = self.tree.query_ball_point(xyz, best_distance + self.max_half_length + 0.1)
        candidates = np.asarray(candidates, dtype=np.int64)
        lower = np.linalg.norm(self.midpoints[candidates] - xyz, axis=1) - self.edges[candidates, 3] / 2
        order = np.argsort(lower)
        for position in order:
            if lower[position] > best_distance + 0.1:
                break
            index = int(candidates[position])
            if index == seed:
                continue
            distance, along = self.closest_on_edge(index, lon, lat)
            # Совпадение у общей вершины разрешается стабильным порядком частей.
            old_rank, rank = int(self.edges[best_edge, 5]), int(self.edges[index, 5])
            if distance < best_distance - 1e-6 or (abs(distance - best_distance) <= 1e-6 and rank < old_rank):
                best_edge, best_distance, best_s = index, distance, along
        row = self.edges[best_edge]
        rank = int(row[5])
        segment = self.segments[rank]
        chainage = min(segment["length"], max(0.0, float(row[4]) + best_s))
        return rank, chainage, best_distance


def geodesic_assign(source_rows, coastline_geojson, metric_crs, max_snap_distance_m, strict_snap):
    import geopandas as gpd
    from src.waves.local_moran.calc import PreparedGeometry
    frame = gpd.read_file(coastline_geojson)
    require(not frame.empty and frame.crs is not None, "Берег пуст или у него нет CRS.")
    input_crs = frame.crs.to_string()
    frame = frame.to_crs(4326)
    def parts():
        for feature_id, geometry in enumerate(frame.geometry):
            for part_id, line in enumerate(iter_line_parts(geometry)):
                yield f"feature_{feature_id:06d}_part_{part_id:04d}", line.coords
    index = GeodesicCoastIndex(parts())
    rows = []
    for number, original in enumerate(source_rows, 1):
        rank, chainage, distance = index.nearest(float(original["lon"]), float(original["lat"]))
        segment = index.segments[rank]
        rows.append({**original, "segment_id": segment["id"], "segment_rank": rank,
                     "chainage_m": chainage, "segment_length_m": segment["length"],
                     "snap_distance_m": distance, "snap_ok": max_snap_distance_m is None or distance <= max_snap_distance_m,
                     "is_closed": segment["closed"]})
        if number % 250 == 0 or number == len(source_rows):
            logger.info("Геодезическая привязка: {}/{}", number, len(source_rows))
    rows.sort(key=lambda row: (row["segment_id"], row["chainage_m"], row["point_id"]))
    previous, order = None, 0
    for row in rows:
        if row["segment_id"] != previous:
            previous, order = row["segment_id"], 0
        row["order_in_segment"] = order
        order += 1
    bad = [row for row in rows if not row["snap_ok"]]
    if bad:
        logger.warning("Слишком далеко от берега: {} точек; примеры={}", len(bad),
                       [(r["point_id"], round(r["snap_distance_m"], 2)) for r in bad[:10]])
        require(not strict_snap, "STRICT_SNAP=True: найдены точки вне допустимого snap-расстояния.")
    distances = [row["snap_distance_m"] for row in rows]
    return PreparedGeometry(point_rows=rows, coastline_crs=input_crs, metric_crs=GEODESIC_METHOD,
                            n_total=len(rows), n_snap_ok=len(rows)-len(bad), n_snap_bad=len(bad),
                            mean_snap_distance_m=sum(distances)/len(distances),
                            max_snap_distance_found_m=max(distances))


def check_database(path):
    from src.waves.activity_indices.constants import SCHEMA_VERSION
    con = readonly(path)
    try:
        require(con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION, "Неверная версия БД индексов.")
        for row in con.execute("PRAGMA quick_check"):
            require(row[0] == "ok", f"Ошибка SQLite: {row[0]}")
        require(con.execute("PRAGMA foreign_key_check").fetchone() is None, "Нарушены внешние ключи.")
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require({"meta", "points", "point_trends"} <= tables, "Нужна wave_activity_indices.db из шага 5.4.")
        require(con.execute("SELECT COUNT(*) FROM points").fetchone()[0] > 0, "Нет точек.")
        missing = con.execute("SELECT p.point_id FROM points p LEFT JOIN point_trends t USING(point_id) WHERE t.point_id IS NULL LIMIT 10").fetchall()
        require(not missing, f"Нет трендов для точек: {missing}")
        meta = dict(con.execute("SELECT key,value FROM meta"))
        gaps = meta.get("source_missing_bathymetry_point_days")
        if gaps is not None and int(gaps) > 0:
            logger.warning("В исходном расчёте {} точко-дней с пропусками батиметрии; этот этап их не исправляет.", gaps)
        return meta
    finally:
        con.close()


def validate_run(path, run_id):
    con = readonly(path)
    con.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require({"local_moran_runs", "local_moran_point_order", "local_moran_results"} <= tables, "Нет запусков Local Moran; сначала выполни calculate.")
        run = (con.execute("SELECT * FROM local_moran_runs ORDER BY run_id DESC LIMIT 1").fetchone()
               if run_id is None else con.execute("SELECT * FROM local_moran_runs WHERE run_id=?", (run_id,)).fetchone())
        require(run is not None, "Запуск не найден.")
        run = dict(run)
        rid = int(run["run_id"])
        mapping = {int(r["point_id"]): dict(r) for r in con.execute("SELECT * FROM local_moran_point_order WHERE run_id=?", (rid,))}
        require(len(mapping) == run["n_points_total"] and bool(mapping), "Запуск неполон: назначения точек.")
        require(all("snap_ok" in r for r in mapping.values()), "Старая схема без snap_ok; нужен пересчёт.")
        included = {pid for pid,r in mapping.items() if r["snap_ok"]}
        require(len(included) == run["n_points_snap_ok"], "Неверный счётчик snap_ok.")
        expected = {(pid, metric) for pid in included for metric in ("normal_mean_wm", run["alongshore_metric"])}
        results = con.execute("SELECT point_id,metric FROM local_moran_results WHERE run_id=?", (rid,)).fetchall()
        actual = {(int(r[0]), r[1]) for r in results}
        require(actual == expected and len(results) == len(expected) == run["n_results"], "Запуск неполон: результаты метрик.")
        return run, mapping
    finally:
        con.close()


def save_geodesic_metadata(path, run_id):
    con = sqlite3.connect(path)
    try:
        with con:
            con.execute("CREATE TABLE IF NOT EXISTS local_moran_geodesic_meta (run_id INTEGER PRIMARY KEY, config_json TEXT NOT NULL)")
            con.execute("INSERT INTO local_moran_geodesic_meta VALUES (?,?)", (run_id, json.dumps({
                "method": GEODESIC_METHOD, "ellipsoid": "WGS84", "max_index_edge_m": MAX_INDEX_EDGE_M,
                "snap_optimization_tolerance_m": SNAP_OPTIMIZATION_TOLERANCE_M,
                "topology": "each original LineString part separately; no automatic merging",
            }, allow_nan=False)))
    finally:
        con.close()


def check_numbers(value):
    if isinstance(value, float):
        require(math.isfinite(value), "В GeoJSON есть неконечное число.")
    elif isinstance(value, dict):
        for item in value.values():
            check_numbers(item)
    elif isinstance(value, list):
        for item in value:
            check_numbers(item)


def reject_constant(value):
    raise ValueError(f"Недопустимое JSON-число: {value}")


def export_checked(database, output, selected_id):
    from src.waves.local_moran.export import export_local_moran_geojson
    run, mapping = validate_run(database, selected_id)
    with TemporaryDirectory(prefix=f".{output.stem}.export-", dir=output.parent) as directory:
        temporary = Path(directory) / "validated.geojson"
        export_local_moran_geojson(database, temporary, run_id=run["run_id"])
        data = json.loads(temporary.read_text(encoding="utf-8"), parse_constant=reject_constant)
        require(data.get("type") == "FeatureCollection", "Ожидался FeatureCollection.")
        features = data.get("features", [])
        require(len(features) == len(mapping), "Неверное число Feature.")
        seen = set()
        for feature in features:
            props = feature["properties"]
            pid = int(props["point_id"])
            require(pid in mapping and pid not in seen, "Неверные или повторные point_id.")
            seen.add(pid)
            feature["id"] = pid
            props["snap_ok"] = int(bool(mapping[pid]["snap_ok"]))
            if not props["snap_ok"]:
                for prefix in ("normal", "alongshore"):
                    for field in ("status", "cluster", "cluster_fdr"):
                        props[f"{prefix}_{field}"] = "SNAP_EXCLUDED"
            geometry = feature.get("geometry", {})
            xy = geometry.get("coordinates", [])
            require(geometry.get("type") == "Point" and len(xy) == 2
                    and all(type(v) in (int,float) and math.isfinite(v) for v in xy)
                    and -180 <= xy[0] <= 180 and -90 <= xy[1] <= 90, "Некорректная геометрия WGS84.")
        data.pop("crs", None)
        data["metadata"] = {"export_created_at": datetime.now(UTC).isoformat(), "run": run,
                            "indices_metadata": check_database(database)}
        check_numbers(data)
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=GEOJSON_INDENT)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    logger.success("GeoJSON: run_id={}, features={}, файл={}", run["run_id"], len(features), output)


def main():
    logger.remove()
    logger.add(sys.stderr, level=LOG_LEVEL, diagnose=False, backtrace=False)
    validate_settings()
    database, output = resolve_path(INDEX_DB), resolve_path(OUTPUT_GEOJSON)
    if not database.is_file():
        raise FileNotFoundError(database)
    calculate = MODE in {"calculate", "calculate_export"}
    export = MODE in {"export", "calculate_export"}
    coastline = resolve_coastline() if calculate else None
    if export:
        output.parent.mkdir(parents=True, exist_ok=True)
        require(output.suffix.lower() in {".json", ".geojson"}, "Выход должен быть GeoJSON.")
        protected = [database] + [Path(f"{database}{s}") for s in ("-wal","-shm","-journal",".lock")]
        if coastline is not None:
            protected.append(coastline)
        for path in protected:
            for candidate in (output, Path(f"{output}.lock")):
                same = candidate == path
                if candidate.exists() and path.exists():
                    same = same or candidate.samefile(path)
                require(not same, "Выход совпадает с входным или служебным файлом.")
    with ExitStack() as stack:
        locks = {Path(f"{database}.lock")}
        if export:
            locks.add(Path(f"{output}.lock"))
        for lock in sorted(locks, key=str):
            stack.enter_context(file_lock(lock))
        if export and output.exists() and not OVERWRITE_GEOJSON:
            raise FileExistsError(f"Файл существует: {output}. Для замены задай OVERWRITE_GEOJSON=True.")
        watcher = readonly(database)
        stack.callback(watcher.close)
        version = watcher.execute("PRAGMA data_version").fetchone()[0]
        state = file_state(database)
        with TemporaryDirectory(prefix=".local_moran-", dir=database.parent) as directory:
            working = Path(directory) / "working.db"
            copy_database(database, working)
            check_database(working)
            selected_id = EXPORT_RUN_ID
            if calculate:
                require(coastline != database, "Береговой файл совпадает с БД.")
                coast_state, coast_hash = file_state(coastline), sha256(coastline)
                module = importlib.import_module("src.waves.local_moran.service")
                # Заменяется только подготовка геометрии в этом процессе,
                # только на время синхронного вызова; затем символ восстанавливается.
                with patch.object(module, "assign_points_to_coastline", geodesic_assign):
                    stats = module.LocalMoranService(
                        working, coastline, radius_m=RADIUS_M, window_width_m=WINDOW_WIDTH_M,
                        alongshore_mode=ALONGSHORE_MODE, metric_crs=None,
                        max_snap_distance_m=MAX_SNAP_DISTANCE_M, strict_snap=STRICT_SNAP,
                        weight_mode=WEIGHT_MODE, permutations=PERMUTATIONS, seed=RANDOM_SEED,
                        alpha_raw=ALPHA_RAW, alpha_fdr=ALPHA_FDR,
                    ).run()
                selected_id = stats.run_id
                validate_run(working, selected_id)
                save_geodesic_metadata(working, selected_id)
                require(file_state(coastline) == coast_state and sha256(coastline) == coast_hash, "Берег изменился во время расчёта.")
                require(watcher.execute("PRAGMA data_version").fetchone()[0] == version and file_state(database) == state,
                        "Рабочая БД изменена другим приложением; результаты не опубликованы.")
                copy_database(working, database)
                logger.success("Запуск {} сохранён: snap_ok={}, snap_bad={}, results={}, db={}",
                               selected_id, stats.n_points_snap_ok, stats.n_points_snap_bad, stats.n_results, database)
            if export:
                export_checked(working, output, selected_id)


if __name__ == "__main__":
    main()
