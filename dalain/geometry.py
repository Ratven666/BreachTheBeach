from dataclasses import dataclass
import math

import geopandas as gpd
import numpy as np

from loguru import logger
from pyproj import CRS, Geod, Transformer

from shapely import (
    STRtree,
    line_merge,
    set_precision,
)

from shapely.geometry import (
    LineString,
    Point,
    Polygon,
    box,
)

from shapely.ops import (
    polygonize,
    transform,
    unary_union,
)


class InvalidScenarioError(ValueError):
    """Недопустимая геометрия текущего кандидата."""


def geometry_parts(geometry, kinds):
    if geometry is None or geometry.is_empty:
        return []

    if geometry.geom_type in kinds:
        return [geometry]

    return [
        part
        for component in getattr(geometry, "geoms", ())
        for part in geometry_parts(component, kinds)
    ]


def neighbours(cell):
    i, j = cell

    return (
        (i - 1, j),
        (i + 1, j),
        (i, j - 1),
        (i, j + 1),
    )


def is_connected(cells):
    if not cells:
        return False

    visited = set()
    pending = [min(cells)]

    while pending:
        cell = pending.pop()

        if cell in visited:
            continue

        visited.add(cell)

        pending.extend(
            other
            for other in neighbours(cell)
            if other in cells
            and other not in visited
        )

    return len(visited) == len(cells)


@dataclass(frozen=True)
class Site:
    x: float
    y: float

    nx: float
    ny: float

    normal_deg: float


@dataclass
class Scenario:
    cells: frozenset
    fill: object
    coast: object
    sites: list

    credited_area_m2: float
    actual_water_area_m2: float

    original_point_count: int = 0
    reclamation_point_count: int = 0


class Geometry:
    def __init__(self, cfg):
        self.cfg = cfg

        # Шаг не больше tolerance_m.
        # Половина стороны квадрата кратна шагу,
        # поэтому сетка квадратов сохраняет геометрию.
        self.precision = (
            cfg.cell_side_m
            / (
                2
                * math.ceil(
                    cfg.cell_side_m
                    / (2 * cfg.tolerance_m)
                )
            )
        )

        self.crs = CRS.from_proj4(
            f"+proj=aeqd "
            f"+lat_0={cfg.center_lat} "
            f"+lon_0={cfg.center_lon} "
            "+datum=WGS84 +units=m"
        )

        self.to_metric = Transformer.from_crs(
            4326,
            self.crs,
            always_xy=True,
        )

        self.to_lonlat = Transformer.from_crs(
            self.crs,
            4326,
            always_xy=True,
        )

        self.geod = Geod(ellps="WGS84")

        radius = max(
            cfg.search_radius_m,
            cfg.evaluation_radius_m,
        )

        margin = max(
            10 * cfg.cell_side_m,
            4 * cfg.tangent_delta_m,
            cfg.point_spacing_m,
        )

        self.frame = self._q(
            box(
                -radius - margin,
                -radius - margin,
                radius + margin,
                radius + margin,
            )
        )

        self.roi = self._q(
            Point(0, 0).buffer(
                cfg.evaluation_radius_m,
                quad_segs=256,
            )
        )

        source = gpd.read_file(
            cfg.coastline_path
        )

        if source.crs is None:
            raise ValueError(
                "У береговой линии нет CRS"
            )

        self.lines = []

        for geometry in source.to_crs(self.crs).geometry:
            if geometry is None or geometry.is_empty:
                continue

            if (
                geometry.geom_type
                not in {"LineString", "MultiLineString"}
                or not geometry.is_valid
            ):
                raise ValueError(
                    "Некорректная направленная береговая линия"
                )

            normalized = self._q(
                geometry
            )

            self.lines.extend(
                line
                for line in geometry_parts(
                    normalized,
                    {"LineString"},
                )
                if line.length > self.precision
            )

        if not self.lines:
            raise ValueError(
                "Пустой берег после приведения точности"
            )

        self.line_tree = STRtree(
            self.lines
        )

        self.original_coast = self._q(
            unary_union(self.lines)
        )

        self.land = None

        if cfg.land_path is not None:
            data = gpd.read_file(
                cfg.land_path
            )

            if data.crs is None:
                raise ValueError(
                    "У маски суши нет CRS"
                )

            polygons = list(
                data.to_crs(self.crs).geometry
            )

            if not polygons or any(
                geometry is None
                or not geometry.is_valid
                or geometry.geom_type
                not in {"Polygon", "MultiPolygon"}
                for geometry in polygons
            ):
                raise ValueError(
                    "Некорректная маска суши"
                )

            self.land = self._q(
                unary_union(polygons)
            )

            self.local_land = self._q(
                self.land.intersection(
                    self.frame
                )
            )

            self.fetch_obstacles = self._q(
                unary_union(
                    [
                        self.original_coast,
                        self.land.boundary,
                    ]
                )
            )

        else:
            self.fetch_obstacles = (
                self.original_coast
            )

            self.local_land = (
                self._local_land()
            )

        self.base_coast = self._q(
            self.local_land.boundary.difference(
                self.frame.boundary
            )
        )

        self.baseline_lakes = (
            self._closed_water(
                self.local_land
            )
        )

        self.base_sites = self._sample(
            self._q(
                self.base_coast.intersection(
                    self.roi
                )
            ),
            self.local_land,
        )

        if not self.base_sites:
            raise ValueError(
                "В радиусе нет исходных береговых точек"
            )

        self.cells = {}
        self.units = {}
        self.water_areas = {}

        self._build_grid()

        self.seeds = sorted(
            cell
            for cell in self.cells
            if (
                cfg.connection == "island"
                or self.units[cell] == 1
            )
        )

        if (
            not self.seeds
            or sum(self.units.values())
            < cfg.target_units
        ):
            raise ValueError(
                "Недостаточно допустимой площади "
                "или нет стартовых квадратов"
            )

        logger.info(
            "Точность геометрии={} м | "
            "исходных точек={} | "
            "допустимых ячеек={}",
            self.precision,
            len(self.base_sites),
            len(self.cells),
        )

    # ========================================================
    # ОБЩИЕ ОПЕРАЦИИ
    # ========================================================

    def _q(self, geometry):
        return set_precision(
            geometry,
            self.precision,
            mode="valid_output",
        )

    def wgs84(self, geometry):
        return transform(
            self.to_lonlat.transform,
            geometry,
        )

    @staticmethod
    def _segments(line):
        coordinates = np.asarray(
            line.coords,
            dtype=float,
        )[:, :2]

        differences = np.diff(
            coordinates,
            axis=0,
        )

        lengths = np.hypot(
            differences[:, 0],
            differences[:, 1],
        )

        cumulative = np.r_[
            0.0,
            np.cumsum(lengths),
        ]

        return (
            coordinates,
            differences,
            lengths,
            cumulative,
        )

    # ========================================================
    # ИСХОДНЫЕ СУША И ВОДА
    # ========================================================

    def _signed_water(self, point):
        line = self.lines[
            int(
                self.line_tree.nearest(
                    point
                )
            )
        ]

        chainage = line.project(point)
        projection = line.interpolate(chainage)

        (
            coordinates,
            differences,
            lengths,
            cumulative,
        ) = self._segments(line)

        index = max(
            0,
            min(
                int(
                    np.searchsorted(
                        cumulative,
                        chainage,
                        side="right",
                    ) - 1
                ),
                len(lengths) - 1,
            ),
        )

        if lengths[index] <= 1e-12:
            raise ValueError(
                "Нулевой сегмент исходного берега"
            )

        tx, ty = (
            differences[index]
            / lengths[index]
        )

        if self.cfg.sea_side == "right":
            nx, ny = ty, -tx
        else:
            nx, ny = -ty, tx

        signed_distance = (
            (point.x - projection.x) * nx
            + (point.y - projection.y) * ny
        )

        return signed_distance > 0

    def _local_land(self):
        coast = self._q(
            self.original_coast.intersection(
                self.frame
            )
        )

        network = self._q(
            unary_union(
                [
                    self.frame.boundary,
                    coast,
                ]
            )
        )

        faces = list(
            polygonize(network)
        )

        if len(faces) < 2:
            raise ValueError(
                "Берег не разделяет локальную область. "
                "Нужна полная локальная линия или land_path."
            )

        land = self._q(
            unary_union(
                [
                    face
                    for face in faces
                    if not self._signed_water(
                        face.representative_point()
                    )
                ]
            )
        )

        if (
            land.is_empty
            or self.frame.difference(land).is_empty
        ):
            raise ValueError(
                "Не удалось классифицировать локальную сушу"
            )

        unmatched_length = (
            coast.difference(
                land.boundary
            ).length
        )

        if unmatched_length > max(
            self.precision,
            coast.length * 1e-9,
        ):
            raise ValueError(
                "Локальный берег содержит разрывы "
                "или несогласованные компоненты. "
                "Задайте land_path или исправьте исходные линии."
            )

        return land

    def original_water(self, point):
        if self.land is not None:
            return not self.land.covers(point)

        if self.frame.contains(point):
            return not self.local_land.covers(point)

        return self._signed_water(point)

    def scenario_water(self, point, fill):
        if (
            not fill.is_empty
            and fill.covers(point)
        ):
            return False

        return self.original_water(point)

    # ========================================================
    # СЕТКА
    # ========================================================

    def _build_grid(self):
        cfg = self.cfg
        side = cfg.cell_side_m

        count = math.ceil(
            cfg.search_radius_m / side
        )

        candidates = [
            (i, j)
            for i in range(-count, count + 1)
            for j in range(-count, count + 1)
            if math.hypot(
                (abs(i) + 0.5) * side,
                (abs(j) + 0.5) * side,
            ) <= cfg.search_radius_m
        ]

        if len(candidates) > cfg.max_grid_cells:
            raise ValueError(
                f"Ячеек в круге: {len(candidates)}; "
                f"лимит: {cfg.max_grid_cells}"
            )

        epsilon_area = max(
            self.precision**2,
            side**2 * 1e-10,
        )

        for i, j in candidates:
            cell = self._q(
                box(
                    (i - 0.5) * side,
                    (j - 0.5) * side,
                    (i + 0.5) * side,
                    (j + 0.5) * side,
                )
            )

            water_area = self._q(
                cell.difference(
                    self.local_land
                )
            ).area

            if water_area <= epsilon_area:
                continue

            crossing = (
                cell.area - water_area
                > epsilon_area
            )

            if cfg.connection == "island":
                if (
                    crossing
                    or cell.intersects(
                        self.base_coast
                    )
                ):
                    continue

            key = (i, j)

            self.cells[key] = cell

            self.units[key] = (
                1 if crossing else 2
            )

            self.water_areas[key] = float(
                water_area
            )

    # ========================================================
    # ТОПОЛОГИЯ
    # ========================================================

    def _closed_water(self, land):
        water = self._q(
            self.frame.difference(land)
        )

        return self._q(
            unary_union(
                [
                    polygon
                    for polygon in geometry_parts(
                        water,
                        {"Polygon"},
                    )
                    if not polygon.intersects(
                        self.frame.boundary
                    )
                ]
            )
        )

    def _fill(self, cells):
        if not cells:
            return self._q(
                Polygon()
            )

        return self._q(
            unary_union(
                [
                    self.cells[cell]
                    for cell in sorted(cells)
                ]
            )
        )

    def _topology(self, cells):
        fill = self._fill(cells)

        if any(
            polygon.interiors
            for polygon in geometry_parts(
                fill,
                {"Polygon"},
            )
        ):
            raise InvalidScenarioError(
                "Замкнутая пустая полость в намыве"
            )

        land = self._q(
            self.local_land.union(fill)
        )

        new_lakes = self._q(
            self._closed_water(land).difference(
                self.baseline_lakes
            )
        )

        if (
            new_lakes.area
            > self.precision**2
        ):
            raise InvalidScenarioError(
                "Намыв перекрыл выход воды "
                "и создал замкнутый водоём"
            )

        return fill, land

    def valid(self, cells):
        if len(cells) < 2:
            return False

        if not set(cells).issubset(self.cells):
            return False

        if not is_connected(cells):
            return False

        area_units = sum(
            self.units[cell]
            for cell in cells
        )

        if area_units != self.cfg.target_units:
            return False

        if self.cfg.connection == "attached":
            if not any(
                self.units[cell] == 1
                for cell in cells
            ):
                return False

        try:
            self._topology(cells)
            return True

        except InvalidScenarioError:
            return False

    # ========================================================
    # КОНТРОЛЬ НОРМАЛЕЙ
    # ========================================================

    def _normal_valid(self, site, land):
        """Нормаль должна разделять сушу и воду
        одновременно на нескольких масштабах."""
        for epsilon in (
            self.cfg.probe_m,
            self.cfg.probe_m * 0.1,
            self.cfg.probe_m * 0.01,
        ):
            water_probe = Point(
                site.x + site.nx * epsilon,
                site.y + site.ny * epsilon,
            )

            land_probe = Point(
                site.x - site.nx * epsilon,
                site.y - site.ny * epsilon,
            )

            if land.covers(water_probe):
                return False

            if not land.covers(land_probe):
                return False

        return True

    def _site(self, line, chainage, land):
        (
            coordinates,
            differences,
            lengths,
            cumulative,
        ) = self._segments(line)

        index = max(
            0,
            min(
                int(
                    np.searchsorted(
                        cumulative,
                        chainage,
                        side="right",
                    ) - 1
                ),
                len(lengths) - 1,
            ),
        )

        if lengths[index] <= 1e-12:
            raise InvalidScenarioError(
                "Нулевая грань берега"
            )

        # Не ставим точку вплотную к углу.
        # Смещение не выходит за выбранную грань.
        gap = min(
            max(
                4 * self.precision,
                self.cfg.probe_m * 0.1,
            ),
            lengths[index] * 0.1,
        )

        chainage = float(
            np.clip(
                chainage,
                cumulative[index] + gap,
                cumulative[index + 1] - gap,
            )
        )

        point = line.interpolate(
            chainage
        )

        tx, ty = (
            differences[index]
            / lengths[index]
        )

        delta = min(
            self.cfg.tangent_delta_m,
            line.length / 4,
        )

        before = line.interpolate(
            max(
                0,
                chainage - delta,
            )
        )

        after = line.interpolate(
            min(
                line.length,
                chainage + delta,
            )
        )

        dx = after.x - before.x
        dy = after.y - before.y

        normal_candidates = []

        norm = math.hypot(dx, dy)

        if norm > 1e-12:
            normal_candidates.append(
                (dy / norm, -dx / norm)
            )

        # Если сглаженная касательная не подходит,
        # используем точную касательную текущей грани.
        normal_candidates.append(
            (float(ty), float(-tx))
        )

        for nx, ny in normal_candidates:
            for sign in (1, -1):
                signed_nx = sign * nx
                signed_ny = sign * ny

                candidate = Site(
                    point.x,
                    point.y,
                    signed_nx,
                    signed_ny,
                    0.0,
                )

                if not self._normal_valid(
                    candidate,
                    land,
                ):
                    continue

                lon, lat = (
                    self.to_lonlat.transform(
                        point.x,
                        point.y,
                    )
                )

                lon2, lat2 = (
                    self.to_lonlat.transform(
                        point.x + signed_nx,
                        point.y + signed_ny,
                    )
                )

                azimuth, _, _ = self.geod.inv(
                    lon,
                    lat,
                    lon2,
                    lat2,
                )

                return Site(
                    point.x,
                    point.y,
                    signed_nx,
                    signed_ny,
                    azimuth % 360,
                )

        raise InvalidScenarioError(
            "Нормаль не разделяет сушу и воду: "
            f"x={point.x:.6f}, y={point.y:.6f}"
        )

    # ========================================================
    # РАССТАНОВКА ТОЧЕК
    # ========================================================

    def _sample(self, coast, land):
        lines = geometry_parts(
            self._q(coast),
            {"LineString"},
        )

        if not lines:
            return []

        merged = line_merge(
            self._q(
                unary_union(lines)
            )
        )

        sites = []
        seen = set()

        for line in geometry_parts(
            merged,
            {"LineString"},
        ):
            if line.length <= self.precision:
                continue

            distances = np.arange(
                self.cfg.point_spacing_m / 2,
                line.length,
                self.cfg.point_spacing_m,
            )

            if not len(distances):
                distances = [
                    line.length / 2
                ]

            for distance in distances:
                site = self._site(
                    line,
                    float(distance),
                    land,
                )

                key = (
                    round(site.x, 7),
                    round(site.y, 7),
                )

                if key not in seen:
                    seen.add(key)
                    sites.append(site)

        return sites

    # ========================================================
    # СЦЕНАРИЙ ТЕКУЩЕГО КАНДИДАТА
    # ========================================================

    def scenario(self, cells):
        cells = frozenset(cells)

        if cells and not self.valid(cells):
            raise InvalidScenarioError(
                "Неверная площадь, связность "
                "или замкнутый водоём"
            )

        fill, land = self._topology(
            cells
        )

        coast = self._q(
            land.boundary
            .difference(
                self.frame.boundary
            )
            .intersection(
                self.roi
            )
        )

        original_sites = []

        for original_site in self.base_sites:
            site = original_site

            point = Point(
                site.x,
                site.y,
            )

            if (
                not fill.is_empty
                and fill.covers(point)
            ):
                continue

            if (
                coast.distance(point)
                > self.precision
            ):
                continue

            # Старая нормаль тоже перепроверяется:
            # рядом мог появиться новый участок суши.
            if not self._normal_valid(site, land):
                lines = geometry_parts(
                    coast,
                    {"LineString"},
                )

                nearest = min(
                    lines,
                    key=lambda line: line.distance(point),
                )

                site = self._site(
                    nearest,
                    nearest.project(point),
                    land,
                )

            original_sites.append(site)

        added_coast = self._q(
            land.boundary
            .difference(
                self.local_land.boundary
            )
            .intersection(
                self.roi
            )
        )

        reclamation_sites = self._sample(
            added_coast,
            land,
        )

        sites = []
        seen = set()

        original_count = 0
        reclamation_count = 0

        for group, group_sites in (
            (0, original_sites),
            (1, reclamation_sites),
        ):
            for site in group_sites:
                key = (
                    round(site.x, 7),
                    round(site.y, 7),
                )

                if key in seen:
                    continue

                if not self._normal_valid(
                    site,
                    land,
                ):
                    raise InvalidScenarioError(
                        "Некорректная нормаль "
                        "после объединения точек"
                    )

                seen.add(key)
                sites.append(site)

                if group == 0:
                    original_count += 1
                else:
                    reclamation_count += 1

        sites.sort(
            key=lambda site: (
                site.x,
                site.y,
            )
        )

        if not sites:
            raise InvalidScenarioError(
                "Нет береговых точек"
            )

        logger.info(
            "ТОЧКИ КАНДИДАТА | "
            "исходных={} | новых={} | всего={} | "
            "берег={:.1f} м",
            original_count,
            reclamation_count,
            len(sites),
            coast.length,
        )

        return Scenario(
            cells=cells,
            fill=fill,
            coast=coast,
            sites=sites,
            credited_area_m2=(
                sum(
                    self.units[cell]
                    for cell in cells
                )
                * self.cfg.cell_side_m**2
                / 2
            ),
            actual_water_area_m2=sum(
                self.water_areas[cell]
                for cell in cells
            ),
            original_point_count=original_count,
            reclamation_point_count=reclamation_count,
        )