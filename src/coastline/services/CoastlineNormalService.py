from __future__ import annotations

import math
from dataclasses import dataclass

import geopandas as gpd
from loguru import logger
from shapely.geometry import LineString, MultiLineString, Point

from src.coastline.domain.CoastlineDataset import CoastlineDataset
from src.coastline.domain.CoastlineNormalPointSet import CoastlineNormalPointSet
from src.coastline.domain.CoastlinePointSet import CoastlinePointSet


@dataclass(frozen=True)
class CoastlineNormalConfig:
    """
    Параметры вычисления нормалей.

    sea_side:
        "left" или "right" относительно направления вершин линии
        в исходном файле. Программа направление НЕ меняет.

    tangent_delta_m:
        Половина интервала, на котором оценивается касательная.

    max_reference_distance_m:
        Максимальное расстояние от точки до референсной линии.
        None отключает проверку.
    """

    sea_side: str = "right"
    normal_length_m: float = 200.0
    tangent_delta_m: float = 5.0
    working_crs: str | None = None
    max_reference_distance_m: float | None = 500.0

    def __post_init__(self) -> None:
        side = self.sea_side.lower()
        if side not in {"left", "right"}:
            raise ValueError("sea_side must be 'left' or 'right'")
        if self.normal_length_m <= 0:
            raise ValueError("normal_length_m must be > 0")
        if self.tangent_delta_m <= 0:
            raise ValueError("tangent_delta_m must be > 0")
        if (
            self.max_reference_distance_m is not None
            and self.max_reference_distance_m < 0
        ):
            raise ValueError("max_reference_distance_m must be >= 0 or None")
        object.__setattr__(self, "sea_side", side)


class CoastlineNormalService:
    """
    Вычисляет нормали к ближайшему компоненту референсной береговой линии.

    1. Линия и точки переводятся в метрическую CRS.
    2. Каждая линия используется в исходном направлении вершин:
       никакого linemerge, reverse или выравнивания по порядку точек.
    3. Для каждой точки выбирается ближайший компонент.
    4. Касательная оценивается вокруг проекции точки на компонент.
    5. Нормаль — касательная, повёрнутая на 90 градусов в сторону sea_side.
    """

    def __init__(self, config: CoastlineNormalConfig | None = None) -> None:
        self.config = config or CoastlineNormalConfig()
        self._log = logger.bind(cls=self.__class__.__name__)

    # ── Публичный API ─────────────────────────────────────────────────

    def build_points_with_normals(
        self,
        point_set: CoastlinePointSet,
        dataset: CoastlineDataset,
        name: str | None = None,
    ) -> CoastlineNormalPointSet:
        coastline_parts, work_crs = self._prepare_main_lines(dataset)
        points = self._prepare_points(point_set, work_crs)

        records: list[dict] = []

        for position, (_, row) in enumerate(points.iterrows()):
            point: Point = row.geometry
            attrs = row.drop(labels=["geometry"]).to_dict()

            part_id, coastline, reference_distance = self._nearest_part(
                point, coastline_parts
            )

            max_distance = self.config.max_reference_distance_m
            if max_distance is not None and reference_distance > max_distance:
                label = attrs.get("seq", attrs.get("point_id", position))
                raise ValueError(
                    f"Point {label!r} is {reference_distance:.3f} m "
                    f"from the nearest reference coastline component; "
                    f"allowed maximum is {max_distance:.3f} m"
                )

            info = self._compute_normal_at_point(point, coastline)

            point_id = attrs.get("point_id", position)
            if point_id is None:
                point_id = position

            records.append(
                {
                    **attrs,
                    "point_id": int(point_id),
                    "reference_part_id": int(part_id),
                    "reference_distance_m": float(reference_distance),
                    "chainage_m": info["chainage_m"],
                    "tangent_delta_used_m": info["tangent_delta_used_m"],
                    "tx": info["tx"],
                    "ty": info["ty"],
                    "nx": info["nx"],
                    "ny": info["ny"],
                    "normal_azimuth_deg": info["normal_azimuth_deg"],
                    "sea_side": self.config.sea_side,
                    "geometry": point,
                }
            )

        self._log_direction_reversals(records)

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=work_crs)

        result = CoastlineNormalPointSet.from_gdf(
            gdf, name=name or f"{point_set.meta.name}_normals"
        )

        self._log.info(
            f"Built {len(result)} normal(s) using "
            f"{len(coastline_parts)} coastline component(s)"
        )
        return result

    def build_normal_lines(
        self,
        point_set: CoastlinePointSet,
        dataset: CoastlineDataset,
        normal_length_m: float | None = None,
    ) -> gpd.GeoDataFrame:
        normals = self.build_points_with_normals(
            point_set=point_set, dataset=dataset
        )
        length = (
            float(normal_length_m)
            if normal_length_m is not None
            else float(self.config.normal_length_m)
        )
        return normals.to_normal_lines_gdf(normal_length_m=length)

    def plot(
        self,
        point_set: CoastlinePointSet,
        dataset: CoastlineDataset,
        figsize: tuple[float, float] = (12, 12),
        coastline_color: str = "black",
        other_color: str = "lightgray",
        point_color: str = "red",
        normal_color: str = "blue",
        linewidth: float = 1.0,
        normal_linewidth: float = 0.8,
    ):
        import matplotlib.pyplot as plt

        _, work_crs = self._prepare_main_lines(dataset)

        main_gdf = dataset.main_gdf.to_crs(work_crs)
        other_gdf = dataset.other_gdf.to_crs(work_crs)
        points = self._prepare_points(point_set, work_crs)
        normal_lines = self.build_normal_lines(point_set, dataset)

        fig, ax = plt.subplots(figsize=figsize)

        if not other_gdf.empty:
            other_gdf.plot(ax=ax, color=other_color, linewidth=linewidth)
        if not main_gdf.empty:
            main_gdf.plot(
                ax=ax, color=coastline_color, linewidth=linewidth * 1.5
            )
        if not normal_lines.empty:
            normal_lines.plot(
                ax=ax, color=normal_color, linewidth=normal_linewidth
            )
        if not points.empty:
            points.plot(ax=ax, color=point_color, markersize=12)

        ax.set_title(f"Normals to coastline ({self.config.sea_side} side)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        return fig, ax

    # ── Подготовка данных ─────────────────────────────────────────────

    def _prepare_main_lines(
        self,
        dataset: CoastlineDataset,
    ) -> tuple[list[LineString], str]:
        work_crs = self.config.working_crs or str(dataset.metric_crs)

        main = dataset.main_gdf.to_crs(work_crs)
        if main.empty:
            raise ValueError("dataset.main_gdf is empty")

        # Направление вершин сохраняется как в исходном файле.
        parts: list[LineString] = []
        for geometry in main.geometry:
            parts.extend(self._extract_lines(geometry))

        parts = [
            line
            for line in parts
            if not line.is_empty
            and len(line.coords) >= 2
            and line.length > 0
        ]
        if not parts:
            raise ValueError(
                "Main coastline has no valid non-empty linear geometries"
            )

        total_length = sum(line.length for line in parts)
        self._log.info(
            f"Reference coastline (directions preserved): "
            f"parts={len(parts)}, total_length={total_length:.3f} m, "
            f"crs={work_crs}"
        )
        return parts, work_crs

    @staticmethod
    def _extract_lines(geometry) -> list[LineString]:
        if geometry is None or geometry.is_empty:
            return []
        if isinstance(geometry, LineString):
            return [geometry]
        if isinstance(geometry, MultiLineString):
            return [
                part
                for part in geometry.geoms
                if part is not None and not part.is_empty
            ]
        return []

    @staticmethod
    def _prepare_points(
        point_set: CoastlinePointSet,
        work_crs: str,
    ) -> gpd.GeoDataFrame:
        if point_set.gdf.empty:
            raise ValueError("CoastlinePointSet is empty")
        if point_set.gdf.crs is None:
            raise ValueError("CoastlinePointSet.gdf has no CRS")

        points = point_set.gdf.to_crs(work_crs).copy()

        invalid = points.geometry.isna() | points.geometry.is_empty
        if invalid.any():
            raise ValueError("Point set contains empty geometries")
        if not all(isinstance(g, Point) for g in points.geometry):
            raise TypeError("All point-set geometries must be Point")

        return points

    @staticmethod
    def _nearest_part(
        point: Point,
        parts: list[LineString],
    ) -> tuple[int, LineString, float]:
        if not parts:
            raise ValueError("Reference coastline contains no parts")

        distances = [float(point.distance(part)) for part in parts]
        part_id = min(range(len(parts)), key=distances.__getitem__)
        return part_id, parts[part_id], distances[part_id]

    # ── Диагностика (только лог, данные не меняются) ──────────────────

    def _log_direction_reversals(self, records: list[dict]) -> None:
        """
        Предупреждает, если точки одной линии, упорядоченные по seq,
        идут то по направлению линии, то против него. Это признак
        участка, где направление линии, возможно, нужно поправить
        в исходных данных.
        """
        by_part: dict[int, list[tuple[float, float]]] = {}
        for position, rec in enumerate(records):
            order = rec.get("seq", position)
            by_part.setdefault(rec["reference_part_id"], []).append(
                (float(order), float(rec["chainage_m"]))
            )

        for part_id, items in by_part.items():
            items.sort()
            diffs = [b[1] - a[1] for a, b in zip(items, items[1:])]
            signs = {d > 0 for d in diffs if d != 0}
            if len(signs) > 1:
                self._log.warning(
                    f"Part {part_id}: points are not monotonic along the "
                    f"line direction; check this line's direction"
                )

    # ── Математика нормали ────────────────────────────────────────────

    def _compute_normal_at_point(
        self,
        point: Point,
        coastline: LineString,
    ) -> dict[str, float]:
        chainage = float(coastline.project(point))

        p0, p1, delta_used = self._tangent_points(coastline, chainage)

        dx = float(p1.x - p0.x)
        dy = float(p1.y - p0.y)

        tangent_norm = math.hypot(dx, dy)
        if tangent_norm <= 1e-12:
            raise ValueError(
                "Failed to compute tangent: selected points coincide"
            )

        tx = dx / tangent_norm
        ty = dy / tangent_norm

        if self.config.sea_side == "left":
            nx, ny = -ty, tx
        else:
            nx, ny = ty, -tx

        normal_norm = math.hypot(nx, ny)
        if normal_norm <= 1e-12:
            raise ValueError("Failed to compute normal vector")

        nx /= normal_norm
        ny /= normal_norm

        # Геодезический азимут: 0 — север, 90 — восток.
        azimuth_deg = (math.degrees(math.atan2(nx, ny)) + 360.0) % 360.0

        dot_product = tx * nx + ty * ny
        if abs(dot_product) > 1e-10:
            raise RuntimeError(
                "Internal error: tangent and normal are not perpendicular; "
                f"dot={dot_product}"
            )

        return {
            "chainage_m": chainage,
            "tangent_delta_used_m": delta_used,
            "tx": tx,
            "ty": ty,
            "nx": nx,
            "ny": ny,
            "normal_azimuth_deg": azimuth_deg,
        }

    # def _tangent_points(
    #     self,
    #     coastline: LineString,
    #     chainage: float,
    # ) -> tuple[Point, Point, float]:
    #     """
    #     Центральная разность по линии; при вырождении интервал
    #     увеличивается, затем применяется односторонняя разность.
    #     Порядок p0 -> p1 всегда совпадает с направлением линии.
    #     """
    #     length = float(coastline.length)
    #     if length <= 0:
    #         raise ValueError("Reference coastline has zero length")
    #
    #     requested_delta = max(float(self.config.tangent_delta_m), 0.01)
    #     delta = min(requested_delta, length)
    #
    #     for _ in range(16):
    #         s0 = max(0.0, chainage - delta)
    #         s1 = min(length, chainage + delta)
    #
    #         if s1 > s0:
    #             p0 = coastline.interpolate(s0)
    #             p1 = coastline.interpolate(s1)
    #             if p0.distance(p1) > 1e-9:
    #                 return p0, p1, delta
    #
    #         if delta >= length:
    #             break
    #         delta = min(length, delta * 2.0)
    #
    #     center = coastline.interpolate(chainage)
    #     delta = min(requested_delta, length)
    #
    #     while delta <= length:
    #         forward_s = min(length, chainage + delta)
    #         if forward_s > chainage:
    #             forward = coastline.interpolate(forward_s)
    #             if center.distance(forward) > 1e-9:
    #                 return center, forward, delta
    #
    #         backward_s = max(0.0, chainage - delta)
    #         if backward_s < chainage:
    #             backward = coastline.interpolate(backward_s)
    #             if backward.distance(center) > 1e-9:
    #                 return backward, center, delta
    #
    #         if delta >= length:
    #             break
    #         delta = min(length, delta * 2.0)
    #
    #     raise ValueError(
    #         "Failed to find a non-zero local segment for tangent calculation"
    #     )

    def _tangent_points(
        self,
        coastline: LineString,
        chainage: float,
    ) -> tuple[Point, Point, float]:
        """
        Выбирает две разные точки для оценки касательной.

        Особенности:
        - сохраняет исходное направление линии;
        - для замкнутых линий использует циклический отсчёт;
        - при совпадении точек уменьшает интервал;
        - при необходимости использует одностороннюю разность;
        - крайний резерв — ближайшее ненулевое ребро.

        Возвращает:
            p0, p1, фактически использованный интервал.
        """
        length = float(coastline.length)

        if not math.isfinite(length) or length <= 0.0:
            raise ValueError(
                f"Reference coastline has invalid length: {length!r}"
            )

        if not math.isfinite(chainage):
            raise ValueError(
                f"Non-finite chainage: {chainage!r}"
            )

        coordinates = list(coastline.coords)

        if len(coordinates) < 2:
            raise ValueError(
                "Reference coastline has fewer than two vertices"
            )

        for index, coordinate in enumerate(coordinates):
            if not (
                math.isfinite(coordinate[0])
                and math.isfinite(coordinate[1])
            ):
                raise ValueError(
                    "Reference coastline contains non-finite "
                    f"coordinates at vertex {index}"
                )

        requested_delta = float(self.config.tangent_delta_m)

        if (
            not math.isfinite(requested_delta)
            or requested_delta <= 0.0
        ):
            raise ValueError(
                "tangent_delta_m must be finite and positive"
            )

        chainage = min(max(float(chainage), 0.0), length)
        closed = bool(coastline.is_closed)

        # Масштабно-зависимый порог: отличаем полезное смещение
        # от погрешности представления координат.
        coordinate_scale = max(
            1.0,
            max(
                abs(value)
                for coordinate in coordinates
                for value in coordinate[:2]
            ),
        )

        coordinate_tolerance = max(
            1e-9,
            8.0 * math.ulp(coordinate_scale),
        )

        def different(p0: Point, p1: Point) -> bool:
            return (
                math.isfinite(p0.x)
                and math.isfinite(p0.y)
                and math.isfinite(p1.x)
                and math.isfinite(p1.y)
                and p0.distance(p1) > coordinate_tolerance
            )

        if closed:
            # Не допускаем охвата всего кольца или двух его
            # половин с совпадающими положениями.
            initial_delta = min(
                requested_delta,
                length / 4.0,
            )

            center_s = chainage % length

            def point_at(distance: float) -> Point:
                return coastline.interpolate(distance % length)

        else:
            initial_delta = min(
                requested_delta,
                length,
            )

            center_s = chainage

            def point_at(distance: float) -> Point:
                return coastline.interpolate(
                    min(max(distance, 0.0), length)
                )

        # Сначала пытаемся сохранить центральную оценку.
        # При вырождении уменьшаем окно, а не увеличиваем
        # его вплоть до полного охвата короткого кольца.
        delta = initial_delta

        for _ in range(32):
            if delta <= 0.0:
                break

            if closed:
                p0 = point_at(center_s - delta)
                p1 = point_at(center_s + delta)

            else:
                s0 = max(0.0, center_s - delta)
                s1 = min(length, center_s + delta)

                p0 = point_at(s0)
                p1 = point_at(s1)

            if different(p0, p1):
                return p0, p1, delta

            delta *= 0.5

        # На возвратном участке центральная разность может
        # выродиться даже при корректной ненулевой геометрии.
        # Пробуем односторонние разности, сохраняя порядок.
        center = point_at(center_s)
        delta = initial_delta

        for _ in range(32):
            if delta <= 0.0:
                break

            if closed or center_s + delta <= length:
                forward = point_at(center_s + delta)

                if different(center, forward):
                    return center, forward, delta

            if closed or center_s - delta >= 0.0:
                backward = point_at(center_s - delta)

                if different(backward, center):
                    return backward, center, delta

            delta *= 0.5

        # Резерв: выбираем ненулевое ребро, ближайшее
        # по расстоянию вдоль линии к искомой позиции.
        #
        # Направление ребра остаётся исходным: vertex[i]
        # -> vertex[i + 1]. Никакого reverse/linemerge.
        best_candidate = None
        best_score = math.inf

        accumulated = 0.0

        for start_coordinate, end_coordinate in zip(
            coordinates,
            coordinates[1:],
        ):
            p0 = Point(
                start_coordinate[0],
                start_coordinate[1],
            )

            p1 = Point(
                end_coordinate[0],
                end_coordinate[1],
            )

            edge_length = float(p0.distance(p1))

            edge_start = accumulated
            edge_end = accumulated + edge_length
            accumulated = edge_end

            if edge_length <= coordinate_tolerance:
                continue

            def distance_to_interval(position: float) -> float:
                if position < edge_start:
                    return edge_start - position

                if position > edge_end:
                    return position - edge_end

                return 0.0

            if closed:
                score = min(
                    distance_to_interval(center_s - length),
                    distance_to_interval(center_s),
                    distance_to_interval(center_s + length),
                )
            else:
                score = distance_to_interval(center_s)

            if score < best_score:
                best_score = score

                best_candidate = (
                    p0,
                    p1,
                    min(requested_delta, edge_length),
                )

        if best_candidate is not None:
            self._log.warning(
                "Tangent fallback: using a non-zero edge; "
                "chainage={:.6f}, length={:.6f}, closed={}",
                chainage,
                length,
                closed,
            )

            return best_candidate

        raise ValueError(
            "Cannot calculate a stable tangent: "
            "no edge exceeds the coordinate precision threshold; "
            f"length={length:.12g}, "
            f"chainage={chainage:.12g}, "
            f"closed={closed}, "
            f"vertices={len(coordinates)}, "
            f"tolerance={coordinate_tolerance:.12g}"
        )
