from __future__ import annotations

"""Один исходный Feature = один комплект CSV/GeoJSON.

Связные части внутри Feature объединяются; несвязные компоненты НЕ
становятся отдельными файлами. Сортировка: component_id, chainage_component_m,
point_id. Разрывы не зашиваются искусственными отрезками.

Зависимости: numpy, pyproj, shapely>=2.0 (GEOS>=3.10).
Разместить файл в Kaliningrad/ или db_pypeline/ проекта.
"""

import csv
import json
import logging
import math
import sqlite3
import tempfile
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point, mapping, shape
from shapely.ops import substring, transform

# ======================== НАСТРОЙКИ ========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Kaliningrad" / "data"
COASTLINES_GEOJSON = DATA_DIR / "coastline" / "T_kaliningrad_merged.geojson"
WAVE_ACTIVITY_DB = DATA_DIR / "db" / "wave_activity.db"
# Новый каталог, чтобы результаты не смешались со старыми 38 комплектами.
OUTPUT_DIR = DATA_DIR / "wave" / "cwef_by_features"

START_DATE = date(1940, 1, 1)  # Фактическое начало полного расчёта.
# Лучше задать явно; None предполагает хотя бы один полный ряд в БД.
END_DATE: date | None = None

INPUT_CRS = "EPSG:4326"  # Если GeoJSON не содержит crs.
WORKING_CRS: str | None = "EPSG:32634"
ENDPOINT_JOIN_TOLERANCE_M = 1.0
MAX_DISTANCE_TO_LINE_M = 5.0
DISTANCE_TIE_TOLERANCE_M = 0.001
# Одна точка -> один Feature. При равных расстояниях:
# first: первый Feature исходного GeoJSON; error: остановиться.
AMBIGUOUS_POINT_POLICY = "first"
# Аналогично внутри одного Feature для совпадающих компонентов.
AMBIGUOUS_COMPONENT_POLICY = "first"

# Компоненты нумеруются внутри Feature по самой ранней входящей части.
# Для открытой компоненты направление согласовано с самой ранней частью,
# но начало ВСЕГДА находится на настоящем конце всей компоненты.
# Для кольца начало по умолчанию — начало самой ранней части.
# Необязательные якоря: (номер Feature, component_id): (lon, lat).
# Открытая компонента: ближайший настоящий конец.
# Кольцо: проекция якоря на кольцо становится нулевым пикетом.
START_ANCHORS_WGS84: dict[tuple[int, int], tuple[float, float]] = {}
REVERSE_COMPONENTS: set[tuple[int, int]] = set()
START_ANCHOR_TOLERANCE_M = 20.0

# strict: отказ при 0 < n_days < длины календаря;
# nan: положительные значения сохранены, остальные ячейки пустые;
# zero: отсутствующие записи принудительно считаются нулями.
# При n_days=0/отсутствии summary столбец всегда пустой.
MISSING_DAY_POLICY = "zero"
CSV_DELIMITER = ","
CSV_ENCODING = "utf-8"
DAY_CHUNK_SIZE = 512
SQL_POINT_BATCH_SIZE = 400
MAX_MATRIX_MIB = 64
OVERWRITE = False
LOG_LEVEL = "INFO"
# ==========================================================

EPOCH = date(1940, 1, 1)
LOG = logging.getLogger("cwef_export")


@dataclass(frozen=True)
class CoastFeature:
    number: int
    source_id: object
    source_line_id: object
    parts: tuple[LineString, ...]


@dataclass(frozen=True)
class Component:
    feature_number: int
    component_id: int
    geometry: LineString  # Метрическая CRS.
    source_parts: tuple[int, ...]
    offset_m: float  # Условная сумма длин предыдущих компонентов, БЕЗ разрывов.
    start_method: str


@dataclass(frozen=True)
class WavePoint:
    point_id: int
    lon: float
    lat: float
    normal_azimuth_deg: int
    n_days: int | None


@dataclass(frozen=True)
class AssignedPoint:
    point: WavePoint
    component: Component
    chainage_m: float
    distance_m: float


@dataclass(frozen=True)
class Stats:
    mean: float | None
    n_values: int
    n_missing: int


def resolve(value: str | Path) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else PROJECT_ROOT / p).resolve()


def validate_settings() -> None:
    for name, value in (("START_DATE", START_DATE), ("END_DATE", END_DATE)):
        if name == "END_DATE" and value is None:
            continue
        if not isinstance(value, date) or isinstance(value, datetime):
            raise TypeError(f"{name}: требуется datetime.date.")
    for name, value in (
        ("ENDPOINT_JOIN_TOLERANCE_M", ENDPOINT_JOIN_TOLERANCE_M),
        ("MAX_DISTANCE_TO_LINE_M", MAX_DISTANCE_TO_LINE_M),
        ("START_ANCHOR_TOLERANCE_M", START_ANCHOR_TOLERANCE_M),
        ("MAX_MATRIX_MIB", MAX_MATRIX_MIB),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} должна быть конечным числом > 0.")
    if not math.isfinite(DISTANCE_TIE_TOLERANCE_M) or DISTANCE_TIE_TOLERANCE_M < 0:
        raise ValueError("DISTANCE_TIE_TOLERANCE_M должна быть >= 0.")
    if MISSING_DAY_POLICY not in {"strict", "nan", "zero"}:
        raise ValueError("Неверная MISSING_DAY_POLICY.")
    if any(p not in {"first", "error"} for p in (
        AMBIGUOUS_POINT_POLICY, AMBIGUOUS_COMPONENT_POLICY
    )):
        raise ValueError("Политики ассоциации: first или error.")
    for value in (DAY_CHUNK_SIZE, SQL_POINT_BATCH_SIZE):
        if not isinstance(value, int) or value < 1:
            raise ValueError("Размеры блоков должны быть целыми > 0.")
    if len(CSV_DELIMITER) != 1:
        raise ValueError("CSV_DELIMITER должен быть одним символом.")
    keys = set(START_ANCHORS_WGS84) | set(REVERSE_COMPONENTS)
    for key in keys:
        if (not isinstance(key, tuple) or len(key) != 2
                or any(not isinstance(v, int) or v < 1 for v in key)):
            raise ValueError("Ключ компоненты: (номер Feature, component_id), с 1.")
    for key, xy in START_ANCHORS_WGS84.items():
        if (len(xy) != 2 or not all(math.isfinite(v) for v in xy)
                or not -180 <= xy[0] <= 180 or not -90 <= xy[1] <= 90):
            raise ValueError(f"Неверный якорь {key}.")


def read_features(path: Path) -> list[CoastFeature]:
    with path.open(encoding="utf-8-sig") as f:
        doc = json.load(f)
    if doc.get("type") == "FeatureCollection":
        features = doc["features"]
    elif doc.get("type") == "Feature":
        features = [doc]
    elif doc.get("type") in {"LineString", "MultiLineString"}:
        features = [{"geometry": doc}]
    else:
        raise ValueError("Ожидался линейный GeoJSON.")
    crs = CRS.from_user_input(INPUT_CRS)
    if doc.get("crs"):
        if doc["crs"].get("type") != "name":
            raise ValueError("Поддерживается GeoJSON crs с type=name.")
        crs = CRS.from_user_input(doc["crs"]["properties"]["name"])
    to_wgs84 = Transformer.from_crs(crs, 4326, always_xy=True)
    result = []
    for number, feature in enumerate(features, 1):
        if feature.get("geometry") is None:
            raise ValueError(f"Feature {number}: пустая геометрия.")
        g = shape(feature["geometry"])
        if isinstance(g, LineString):
            raw_parts = [g]
        elif isinstance(g, MultiLineString):
            raw_parts = list(g.geoms)
        else:
            raise TypeError(f"Feature {number}: неподдерживаемый {g.geom_type}.")
        parts = []
        for part_number, part in enumerate(raw_parts, 1):
            coords = [(float(c[0]), float(c[1])) for c in part.coords]
            if len(coords) < 2 or not all(math.isfinite(v) for c in coords for v in c):
                raise ValueError(f"Feature {number}, часть {part_number}: неверные координаты.")
            line = transform(to_wgs84.transform, LineString(coords))
            if (line.is_empty or not line.is_valid or line.length <= 0
                    or not all(math.isfinite(x) and math.isfinite(y)
                               and -180 <= x <= 180 and -90 <= y <= 90
                               for x, y in line.coords)):
                raise ValueError(f"Feature {number}, часть {part_number}: проверьте CRS/геометрию.")
            parts.append(line)
        if not parts:
            raise ValueError(f"Feature {number}: нет частей.")
        props = feature.get("properties") or {}
        result.append(CoastFeature(number, feature.get("id"), props.get("line_id"), tuple(parts)))
    if not result:
        raise ValueError("В GeoJSON нет объектов.")
    return result


def metric_crs(features: list[CoastFeature]) -> CRS:
    if WORKING_CRS is not None:
        crs = CRS.from_user_input(WORKING_CRS)
    else:
        b = np.array([p.bounds for f in features for p in f.parts])
        lo, la = b[:, :2].min(axis=0)
        hi, ha = b[:, 2:].max(axis=0)
        if hi - lo > 6 or la < -80 or ha > 84 or la < 0 < ha:
            raise ValueError("Задайте WORKING_CRS явно для данного охвата.")
        zone = min(60, max(1, int((((lo + hi) / 2) + 180) // 6) + 1))
        crs = CRS.from_epsg((32600 if (la + ha) / 2 >= 0 else 32700) + zone)
    if (not crs.is_projected or len(crs.axis_info) < 2
            or any(not math.isclose(a.unit_conversion_factor, 1., rel_tol=0, abs_tol=1e-12)
                   for a in crs.axis_info[:2])):
        raise ValueError("WORKING_CRS должна быть проекцией в метрах.")
    return crs


def rotate_ring(line: LineString, distance: float) -> LineString:
    if distance <= 0 or distance >= line.length:
        return line
    tail = substring(line, distance, line.length)
    head = substring(line, 0, distance)
    return LineString([*tail.coords, *list(head.coords)[1:]])


def build_components(features: list[CoastFeature], crs: CRS) -> list[Component]:
    to_metric = Transformer.from_crs(4326, crs, always_xy=True)
    result = []
    for feature in features:
        parts = [transform(to_metric.transform, p) for p in feature.parts]
        if any(not g.is_valid or not math.isfinite(g.length) or g.length <= 0 for g in parts):
            raise ValueError(f"Feature {feature.number}: ошибка проекции.")
        ends = [Point(c) for g in parts for c in (g.coords[0], g.coords[-1])]
        parent = list(range(len(ends)))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        tree = STRtree(ends)
        for i, p in enumerate(ends):
            for raw_j in tree.query(p, predicate="dwithin", distance=ENDPOINT_JOIN_TOLERANCE_M):
                j = int(raw_j)
                if j > i:
                    a, b = find(i), find(j)
                    parent[max(a, b)] = min(a, b)
        clusters = defaultdict(list)
        for i in range(len(ends)):
            clusters[find(i)].append(i)
        node_xy = {}
        for node, members in clusters.items():
            for k, i in enumerate(members):
                for j in members[k + 1:]:
                    if ends[i].distance(ends[j]) > ENDPOINT_JOIN_TOLERANCE_M:
                        raise ValueError(f"Feature {feature.number}: неоднозначный кластер концов.")
            p = ends[min(members)]
            node_xy[node] = (p.x, p.y)
        nodes = [(find(2*i), find(2*i+1)) for i in range(len(parts))]
        adjacency = defaultdict(list)
        for edge, (a, b) in enumerate(nodes):
            adjacency[a].append(edge)
            adjacency[b].append(edge)
        for node, edges in adjacency.items():
            if len(edges) > 2:
                raise ValueError(f"Feature {feature.number}: разветвление у частей "
                                 f"{sorted({e + 1 for e in edges})}. "
                                 "Для разветвлённой сети нет единственного линейного пикетажа.")
        unseen = set(range(len(parts)))
        groups = []
        while unseen:
            stack, group = [min(unseen)], set()
            while stack:
                edge = stack.pop()
                if edge in group:
                    continue
                group.add(edge)
                for node in nodes[edge]:
                    stack.extend(e for e in adjacency[node] if e not in group)
            unseen.difference_update(group)
            groups.append(group)
        groups.sort(key=min)
        offset = 0.
        for component_id, group in enumerate(groups, 1):
            earliest = min(group)
            vertices = {v for e in group for v in nodes[e]}
            endpoints = sorted(v for v in vertices if len(adjacency[v]) == 1)
            if len(endpoints) not in {0, 2}:
                raise ValueError(f"Feature {feature.number}: неверная топология цепочки.")
            closed = not endpoints
            current = nodes[earliest][0] if closed else endpoints[0]
            start = current
            used, traversal = set(), []
            while len(used) < len(group):
                options = sorted({e for e in adjacency[current] if e in group and e not in used})
                if not options:
                    raise ValueError("Обход компоненты оборвался.")
                if not used and closed:
                    edge = earliest
                elif len(options) == 1:
                    edge = options[0]
                else:
                    raise ValueError("Неоднозначный обход компоненты.")
                a, b = nodes[edge]
                forward = current == a
                traversal.append((edge, forward))
                used.add(edge)
                current = b if forward else a
            if (closed and current != start) or (not closed and current not in endpoints):
                raise ValueError("Неверное завершение обхода.")
            if not next(forward for edge, forward in traversal if edge == earliest):
                traversal = [(e, not f) for e, f in reversed(traversal)]
            coords = []
            for edge, forward in traversal:
                a, b = nodes[edge]
                c = list(parts[edge].coords)
                c[0], c[-1] = node_xy[a], node_xy[b]
                if not forward:
                    c.reverse()
                if coords:
                    if coords[-1] != c[0]:
                        raise ValueError("Ошибка стыковки.")
                    c = c[1:]
                for xy in c:
                    if not coords or xy != coords[-1]:
                        coords.append(xy)
            if len(coords) < 2:
                raise ValueError("Вырожденная компонента после привязки концов.")
            line = LineString(coords)
            if not line.is_valid or line.length <= 0 or line.is_closed != closed:
                raise ValueError("Ошибка геометрии собранной компоненты.")
            # Не исправляем внутренние самопересечения молча.
            if not line.is_simple:
                raise ValueError(f"Feature {feature.number}, компонента {component_id}: "
                                 "самопересечение/перекрытие; пикетаж неоднозначен.")
            key = (feature.number, component_id)
            method = "first_source_part_start" if closed else "true_endpoint"
            if key in START_ANCHORS_WGS84:
                anchor = Point(*to_metric.transform(*START_ANCHORS_WGS84[key]))
                if not math.isfinite(anchor.x) or not math.isfinite(anchor.y):
                    raise ValueError(f"Некорректная проекция якоря {key}.")
                if closed:
                    if line.distance(anchor) > START_ANCHOR_TOLERANCE_M:
                        raise ValueError(f"Якорь {key} далеко от кольца.")
                    line = rotate_ring(line, float(line.project(anchor)))
                else:
                    da, db = [anchor.distance(Point(c)) for c in (line.coords[0], line.coords[-1])]
                    if min(da, db) > START_ANCHOR_TOLERANCE_M or abs(da-db) < 1e-6:
                        raise ValueError(f"Якорь {key} не задаёт однозначно настоящий конец.")
                    if db < da:
                        line = LineString(list(line.coords)[::-1])
                        traversal.reverse()
                method = "explicit_anchor"
            if key in REVERSE_COMPONENTS:
                line = LineString(list(line.coords)[::-1])
                traversal.reverse()
                method += "_reversed"
            result.append(Component(feature.number, component_id, line,
                                    tuple(e+1 for e, _ in traversal), offset, method))
            offset += line.length
        LOG.info("Feature %d: исходных частей=%d; компонентов=%d; файлов CWEF=1",
                 feature.number, len(parts), len(groups))
    available = {(c.feature_number, c.component_id) for c in result}
    invalid = (set(START_ANCHORS_WGS84) | set(REVERSE_COMPONENTS)) - available
    if invalid:
        raise ValueError(f"Несуществующие компоненты в настройках: {sorted(invalid)}")
    return result


def read_points(con: sqlite3.Connection) -> list[WavePoint]:
    version = con.execute("PRAGMA user_version").fetchone()[0]
    if version != 3:
        raise ValueError(f"Ожидалась схема wave_activity.db v3, получена v{version}.")
    required = {
        "wave_points": {"id", "lon", "lat", "normal_azimuth_deg"},
        "wave_activity": {"point_id", "day", "cwef_wm"},
        "wave_activity_summary": {"point_id", "n_days"},
    }
    for table, columns in required.items():
        actual = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
        if not columns <= actual:
            raise ValueError(f"{table}: отсутствуют {columns - actual}.")
    points = [WavePoint(*r) for r in con.execute(
        "SELECT p.id,p.lon,p.lat,p.normal_azimuth_deg,s.n_days "
        "FROM wave_points p LEFT JOIN wave_activity_summary s ON s.point_id=p.id ORDER BY p.id"
    )]
    if not points:
        raise ValueError("wave_points пуста.")
    for p in points:
        if not (math.isfinite(p.lon) and math.isfinite(p.lat)
                and -180 <= p.lon <= 180 and -90 <= p.lat <= 90):
            raise ValueError(f"ID={p.point_id}: неверные координаты.")
        if p.n_days is not None and p.n_days < 0:
            raise ValueError(f"ID={p.point_id}: отрицательное n_days.")
    return points


def calendar(con, points):
    if END_DATE is None:
        n = max((p.n_days or 0 for p in points), default=0)
        if not n:
            raise ValueError("Задайте END_DATE явно.")
        end = START_DATE + timedelta(days=n-1)
        LOG.warning("END_DATE=%s по max(n_days)=%d; предполагается хотя бы один полный ряд.", end, n)
    else:
        end = END_DATE
    first, last = (START_DATE-EPOCH).days, (end-EPOCH).days
    if not 0 <= first <= last <= 65535:
        raise ValueError("Неверные даты полного расчётного периода.")
    lo, hi = con.execute("SELECT MIN(day),MAX(day) FROM wave_activity").fetchone()
    if lo is not None and (lo < first or hi > last):
        raise ValueError("Активность выходит за календарь: укажите полный период расчёта.")
    LOG.info("Календарь %s .. %s, дней=%d", START_DATE, end, last-first+1)
    return first, last


def associate(features, components, points, crs):
    tree = STRtree([c.geometry for c in components])
    to_metric = Transformer.from_crs(4326, crs, always_xy=True)
    groups = {f.number: [] for f in features}
    unmatched, ties = [], []
    for p in points:
        x, y = to_metric.transform(p.lon, p.lat)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"ID={p.point_id}: ошибка проекции.")
        xy = Point(x,y)
        candidates = []
        for raw_i in tree.query(xy, predicate="dwithin", distance=MAX_DISTANCE_TO_LINE_M):
            i = int(raw_i)
            candidates.append((float(components[i].geometry.distance(xy)), i))
        if not candidates:
            unmatched.append(p)
            continue
        minimum = min(d for d, _ in candidates)
        tied = [(dist,i) for dist,i in candidates if dist <= minimum + DISTANCE_TIE_TOLERANCE_M]
        feature_numbers = sorted({components[i].feature_number for _,i in tied})
        if len(feature_numbers) > 1:
            if AMBIGUOUS_POINT_POLICY == "error":
                raise ValueError(f"ID={p.point_id}: равные расстояния к Feature {feature_numbers}.")
            ties.append((p.point_id, feature_numbers))
        feature_number = feature_numbers[0]
        options = [(dist,i) for dist,i in tied if components[i].feature_number == feature_number]
        if len(options) > 1 and AMBIGUOUS_COMPONENT_POLICY == "error":
            raise ValueError(f"ID={p.point_id}: неоднозначная компонента внутри Feature {feature_number}.")
        dist, index = min(options, key=lambda v: components[v[1]].component_id)
        c = components[index]
        groups[feature_number].append(AssignedPoint(p,c,float(c.geometry.project(xy)),dist))
    for number, group in groups.items():
        group.sort(key=lambda a: (a.component.component_id,a.chainage_m,a.point.point_id))
        LOG.info("Feature %d: точек=%d",number,len(group))
    LOG.warning("Несопоставленных точек=%d; межобъектных совпадений=%d",len(unmatched),len(ties))
    if not any(groups.values()):
        raise ValueError("Ни одна точка не сопоставлена: проверьте CRS и допуск.")
    return groups, unmatched, ties


def status(p, n):
    if p.n_days is None:
        return "missing_summary"
    if p.n_days == 0:
        return "no_valid_days"
    return "full_count" if p.n_days == n else "incomplete_unknown_dates"


def check_coverage(groups, n):
    for group in groups.values():
        for a in group:
            days = a.point.n_days
            if days is not None and days > n:
                raise ValueError(f"ID={a.point.point_id}: n_days превышает длину календаря.")
            if days and days < n:
                message = f"ID={a.point.point_id}: n_days={days}, календарь={n}; даты пропусков неизвестны."
                if MISSING_DAY_POLICY == "strict":
                    raise ValueError(message + " Проверьте даты либо выберите nan/zero явно.")
                LOG.warning("%s Политика=%s",message,MISSING_DAY_POLICY)


def write_daily(con, path, group, first, last):
    n = last-first+1
    ids = [a.point.point_id for a in group]
    columns = {pid:i for i,pid in enumerate(ids)}
    defaults = np.array([
        0 if (a.point.n_days == n or (bool(a.point.n_days) and MISSING_DAY_POLICY == "zero")) else -1
        for a in group], dtype=np.int64)
    sums, counts = [0]*len(ids), [0]*len(ids)
    chunk = min(DAY_CHUNK_SIZE,max(1,int(MAX_MATRIX_MIB*1024*1024)//(8*max(1,len(ids)))))
    batch_size = min(SQL_POINT_BATCH_SIZE,con.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)-2)
    if batch_size < 1:
        raise ValueError("Недостаточный лимит SQLite.")
    with path.open("w",encoding=CSV_ENCODING,newline="") as stream:
        writer = csv.writer(stream,delimiter=CSV_DELIMITER)
        writer.writerow(["date",*ids])
        for begin in range(first,last+1,chunk):
            end = min(last,begin+chunk-1)
            matrix = np.broadcast_to(defaults,(end-begin+1,len(ids))).copy()
            for offset in range(0,len(ids),batch_size):
                batch = ids[offset:offset+batch_size]
                placeholders = ','.join('?' for _ in batch)
                for pid,day,value in con.execute(
                    f"SELECT point_id,day,cwef_wm FROM wave_activity WHERE point_id IN ({placeholders}) "
                    "AND day BETWEEN ? AND ?",[*batch,begin,end]):
                    col = columns[pid]
                    if not isinstance(value,int) or value < 1 or not group[col].point.n_days:
                        raise ValueError(f"Несогласованные activity/summary: ID={pid}, day={day}.")
                    matrix[day-begin,col] = value
            for row_index, row in enumerate(matrix):
                values = []
                for col, raw in enumerate(row):
                    value = int(raw)
                    if value < 0:
                        values.append("")
                    else:
                        values.append(value)
                        sums[col] += value
                        counts[col] += 1
                writer.writerow([(EPOCH+timedelta(days=begin+row_index)).isoformat(),*values])
    return [Stats(sums[i]/counts[i] if counts[i] else None,counts[i],n-counts[i]) for i in range(len(ids))]


POINT_FIELDS = [
    "point_id","order","feature_number","component_id","component_order",
    "chainage_component_m","chainage_feature_excluding_gaps_m","lon","lat",
    "normal_azimuth_deg","distance_to_line_m","mean_cwef_wm","n_calendar_days",
    "n_values_for_mean","n_missing_days","n_days_in_db","coverage_status",
    "is_closed_component","start_date","end_date","missing_day_policy",
]


def write_points(csv_path, geo_path, feature, group, stats, first, last, crs):
    with (csv_path.open("w",encoding=CSV_ENCODING,newline="") as cs,
          geo_path.open("w",encoding="utf-8",newline="\n") as gs):
        writer = csv.DictWriter(cs,fieldnames=POINT_FIELDS,delimiter=CSV_DELIMITER)
        writer.writeheader()
        gs.write('{"type":"FeatureCollection","metadata":')
        json.dump({"feature_number":feature.number,"source_feature_id":feature.source_id,
                   "source_line_id":feature.source_line_id,"chainage_crs":crs.to_string(),
                   "mean_basis":"exported_values_including_zeros","cwef_units":"W/m",
                   "ordering":"component_id,chainage_component_m,point_id",
                   "feature_chainage_definition":"sum_of_previous_component_lengths_plus_local_chainage; gaps_excluded"},
                  gs,ensure_ascii=False,allow_nan=False)
        gs.write(',"features":[')
        component_orders = defaultdict(int)
        for order,(a,s) in enumerate(zip(group,stats,strict=True),1):
            p,c = a.point,a.component
            component_orders[c.component_id] += 1
            properties = dict(zip(POINT_FIELDS,[
                p.point_id,order,feature.number,c.component_id,component_orders[c.component_id],
                a.chainage_m,c.offset_m+a.chainage_m,p.lon,p.lat,p.normal_azimuth_deg,
                a.distance_m,s.mean,last-first+1,s.n_values,s.n_missing,p.n_days,
                status(p,last-first+1),c.geometry.is_closed,
                (EPOCH+timedelta(days=first)).isoformat(),(EPOCH+timedelta(days=last)).isoformat(),
                MISSING_DAY_POLICY],strict=True))
            writer.writerow(properties)
            if order > 1:
                gs.write(',')
            json.dump({"type":"Feature","id":p.point_id,
                       "geometry":{"type":"Point","coordinates":[p.lon,p.lat]},
                       "properties":properties},gs,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        gs.write(']}\n')


def write_diagnostics(staging, features, components, crs, unmatched, ties):
    to_wgs84 = Transformer.from_crs(crs,4326,always_xy=True)
    fc, endpoints, summary = [], [], []
    for feature in features:
        selected = [c for c in components if c.feature_number == feature.number]
        geometries = [transform(to_wgs84.transform,c.geometry) for c in selected]
        fc.append({"type":"Feature","id":feature.number,
                   "geometry":mapping(MultiLineString(geometries)),
                   "properties":{"feature_number":feature.number,"source_line_id":feature.source_line_id,
                                 "component_count":len(selected)}})
        for c,g in zip(selected,geometries,strict=True):
            summary.append({"feature_number":c.feature_number,"component_id":c.component_id,
                            "source_parts":list(c.source_parts),"length_m":c.geometry.length,
                            "offset_excluding_gaps_m":c.offset_m,"is_closed":c.geometry.is_closed,
                            "start_method":c.start_method})
            for role,xy in [("start",g.coords[0]),("end",g.coords[-1])]:
                endpoints.append({"type":"Feature","geometry":{"type":"Point","coordinates":list(xy)},
                                  "properties":{"feature_number":c.feature_number,
                                                "component_id":c.component_id,"role":role,
                                                "is_closed":c.geometry.is_closed}})
    for name, records in [("coastlines_components.geojson",fc),("component_endpoints.geojson",endpoints)]:
        (staging/name).write_text(json.dumps({"type":"FeatureCollection","features":records},
                                           ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (staging/'export_geometry_report.json').write_text(json.dumps(
        {"feature_count":len(features),"component_count":len(components),"metric_crs":crs.to_string(),
         "components":summary,"ambiguous_point_policy":AMBIGUOUS_POINT_POLICY,
         "ambiguous_feature_points":[{"point_id":pid,"feature_numbers":numbers} for pid,numbers in ties],
         "unassigned_point_count":len(unmatched)},ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    with (staging/'unassigned_points.csv').open('w',encoding=CSV_ENCODING,newline='') as f:
        writer = csv.writer(f,delimiter=CSV_DELIMITER)
        writer.writerow(['point_id','lon','lat'])
        writer.writerows((p.point_id,p.lon,p.lat) for p in unmatched)


def export() -> list[Path]:
    validate_settings()
    geo,db,out = map(resolve,(COASTLINES_GEOJSON,WAVE_ACTIVITY_DB,OUTPUT_DIR))
    for path in (geo,db):
        if not path.is_file():
            raise FileNotFoundError(path)
    features = read_features(geo)
    crs = metric_crs(features)
    components = build_components(features,crs)
    names = {f.number:(f"coastline_{f.number:04d}_cwef.csv",
                       f"coastline_{f.number:04d}_points.csv",
                       f"coastline_{f.number:04d}_points.geojson") for f in features}
    all_names = [n for group in names.values() for n in group] + [
        'coastlines_components.geojson','component_endpoints.geojson',
        'export_geometry_report.json','unassigned_points.csv']
    targets = [out/n for n in all_names]
    def check_targets():
        for target in targets:
            if target.is_dir():
                raise IsADirectoryError(target)
            if target.exists() and not OVERWRITE:
                raise FileExistsError(f"{target}: задайте OVERWRITE=True или новый OUTPUT_DIR.")
    check_targets()
    with closing(sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)) as con:
        con.execute('PRAGMA query_only=ON')
        con.execute('BEGIN')
        points = read_points(con)
        first,last = calendar(con,points)
        groups,unmatched,ties = associate(features,components,points,crs)
        check_coverage(groups,last-first+1)
        out.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.cwef-export-',dir=out) as temporary:
            staging = Path(temporary)
            write_diagnostics(staging,features,components,crs,unmatched,ties)
            for feature in features:
                values_name,points_name,geo_name = names[feature.number]
                group = groups[feature.number]
                stats = write_daily(con,staging/values_name,group,first,last)
                write_points(staging/points_name,staging/geo_name,feature,group,stats,first,last,crs)
            check_targets()
            # Атомарен отдельный replace, а не весь набор файлов.
            for target in targets:
                (staging/target.name).replace(target)
    LOG.info('Готово: Feature=%d, компонентов=%d, комплектов CWEF=%d; %s',
             len(features),len(components),len(features),out)
    return targets


def main() -> None:
    logging.basicConfig(level=LOG_LEVEL,format='%(levelname)s | %(message)s')
    try:
        export()
    except Exception:
        LOG.exception('Ошибка экспорта CWEF')
        raise


if __name__ == '__main__':
    main()
