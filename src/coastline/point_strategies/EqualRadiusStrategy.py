from __future__ import annotations

from collections.abc import Iterable

import geopandas as gpd
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point
from shapely.ops import linemerge

from src.coastline.point_strategies.PointExtractionStrategy import (
    PointExtractionStrategy,
    PointSource,
)


class EqualRadiusStrategy(PointExtractionStrategy):
    """
    Точки, где каждая следующая точка находится на одинаковом
    евклидовом расстоянии от предыдущей.

    Важно:
    - это НЕ "шаг вдоль линии";
    - это "радиус от предыдущей точки";
    - следующая точка находится как пересечение линии
      с окружностью радиуса radius_step_m вокруг предыдущей точки;
    - расчёты выполняются в метрической CRS;
    - результат возвращается в исходную CRS.
    """

    def __init__(
        self,
        radius_step_m: float,
        source: PointSource = PointSource.ALL_LINES,
        include_origin: bool = True,
        include_endpoint: bool = True,
        working_crs: str | None = None,
        input_crs: str | None = None,
        search_tolerance_m: float = 1e-6,
    ) -> None:
        super().__init__(source)

        if radius_step_m <= 0:
            raise ValueError("radius_step_m must be > 0")

        self.radius_step_m = float(radius_step_m)
        self.include_origin = include_origin
        self.include_endpoint = include_endpoint
        self.working_crs = working_crs
        self.input_crs = input_crs
        self.search_tolerance_m = float(search_tolerance_m)

    def _extract_from_lines(self, gdf: gpd.GeoDataFrame) -> list[dict]:
        if gdf.empty:
            return []

        gdf_src = gdf.copy()

        if gdf_src.crs is None:
            if self.input_crs is None:
                raise ValueError(
                    "Input GeoDataFrame has no CRS. "
                    "Pass input_crs to the strategy or set gdf.crs before use."
                )
            gdf_src = gdf_src.set_crs(self.input_crs)

        work_crs = self.working_crs or gdf_src.estimate_utm_crs()
        if work_crs is None:
            raise ValueError("Failed to determine working metric CRS")

        gdf_work = gdf_src.to_crs(work_crs)

        merged_lines = self._collect_merged_lines(gdf_work)
        if not merged_lines:
            return []

        records_work: list[dict] = []

        for line_id, line in merged_lines:
            points = self._build_equal_radius_points(line)

            if not points:
                continue

            start_point = Point(line.coords[0])

            for order, pt in enumerate(points):
                records_work.append(
                    {
                        "line_id": str(line_id),
                        "point_type": "equal_radius",
                        "radius_m": float(start_point.distance(pt)),
                        "point_order": int(order),
                        "geometry": pt,
                    }
                )

        if not records_work:
            return []

        points_work = gpd.GeoDataFrame(
            records_work,
            geometry="geometry",
            crs=work_crs,
        )
        points_out = points_work.to_crs(gdf_src.crs)

        return points_out.to_dict("records")

    def _build_equal_radius_points(self, line: LineString) -> list[Point]:
        coords = list(line.coords)
        if len(coords) < 2 or line.length <= 0:
            return []

        result: list[Point] = []

        current_point = Point(coords[0])
        endpoint = Point(coords[-1])

        if self.include_origin:
            result.append(current_point)

        while True:
            next_point = self._find_next_point_at_radius(
                line=line,
                current_point=current_point,
                radius=self.radius_step_m,
            )

            if next_point is None:
                break

            # защита от зацикливания
            if next_point.distance(current_point) <= self.search_tolerance_m:
                break

            result.append(next_point)
            current_point = next_point

        if self.include_endpoint:
            if not result:
                result.append(endpoint)
            elif result[-1].distance(endpoint) > self.search_tolerance_m:
                result.append(endpoint)

        return result

    def _find_next_point_at_radius(
        self,
        line: LineString,
        current_point: Point,
        radius: float,
    ) -> Point | None:
        """
        Ищет следующую точку на линии, находящуюся на расстоянии radius
        от current_point, причём только ВПЕРЁД по линии.
        """
        current_measure = float(line.project(current_point))
        circle = current_point.buffer(radius)

        intersection = line.intersection(circle.boundary)
        candidate_points = self._extract_points(intersection)

        if not candidate_points:
            return None

        forward_candidates: list[tuple[float, Point]] = []

        for pt in candidate_points:
            measure = float(line.project(pt))

            # берём только точки дальше по ходу линии
            if measure > current_measure + self.search_tolerance_m:
                forward_candidates.append((measure, pt))

        if not forward_candidates:
            return None

        forward_candidates.sort(key=lambda x: x[0])
        return forward_candidates[0][1]

    def _collect_merged_lines(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> list[tuple[str, LineString]]:
        geometries: list[tuple[str, LineString]] = []

        for group_id, lines in self._iter_grouped_lines(gdf):
            merged = self._merge_lines(lines)

            if isinstance(merged, LineString):
                if len(merged.coords) >= 2 and merged.length > 0:
                    geometries.append((group_id, merged))

            elif isinstance(merged, MultiLineString):
                for idx, part in enumerate(merged.geoms):
                    if len(part.coords) >= 2 and part.length > 0:
                        geometries.append((f"{group_id}_{idx}", part))

        return geometries

    def _iter_grouped_lines(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> Iterable[tuple[str, list[LineString]]]:
        lines: list[LineString] = [line for _, line in self._iter_lines(gdf)]

        if not lines:
            return []

        yield "line", lines

    @staticmethod
    def _merge_lines(lines: list[LineString]):
        if not lines:
            return None
        if len(lines) == 1:
            return lines[0]
        return linemerge(lines)

    @staticmethod
    def _extract_points(geom) -> list[Point]:
        if geom is None or geom.is_empty:
            return []

        if isinstance(geom, Point):
            return [geom]

        if isinstance(geom, MultiLineString):
            pts: list[Point] = []
            for part in geom.geoms:
                pts.extend(EqualRadiusStrategy._extract_points(part))
            return pts

        if isinstance(geom, GeometryCollection):
            pts: list[Point] = []
            for part in geom.geoms:
                pts.extend(EqualRadiusStrategy._extract_points(part))
            return pts

        if geom.geom_type == "MultiPoint":
            return [g for g in geom.geoms if isinstance(g, Point)]

        return []
