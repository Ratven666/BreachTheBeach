from __future__ import annotations

from collections.abc import Iterable

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

from src.coastline.point_strategies.PointExtractionStrategy import (
    PointExtractionStrategy,
    PointSource,
)


class EqualStepAlongLineStrategy(PointExtractionStrategy):
    """
    Точки через равный шаг в метрах вдоль ЦЕЛОЙ линии, а не каждого сегмента отдельно.

    Важно:
    - если линия представлена набором сегментов, они сначала объединяются в одну трассу;
    - шаг откладывается по накопленной длине вдоль этой трассы;
    - расчёты выполняются в метрической CRS;
    - результат возвращается в исходную CRS.
    """

    def __init__(
        self,
        step_m: float,
        source: PointSource = PointSource.ALL_LINES,
        include_endpoints: bool = True,
        working_crs: str | None = None,
        input_crs: str | None = None,
    ) -> None:
        super().__init__(source)

        if step_m <= 0:
            raise ValueError("step_m must be > 0")

        self.step_m = float(step_m)
        self.include_endpoints = include_endpoints
        self.working_crs = working_crs
        self.input_crs = input_crs

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

        for line_id, merged_line in merged_lines:
            length_m = float(merged_line.length)
            if length_m <= 0:
                continue

            distances_m: list[float] = []
            d = 0.0
            while d <= length_m:
                distances_m.append(d)
                d += self.step_m

            if self.include_endpoints:
                if not distances_m:
                    distances_m = [0.0, length_m]
                elif distances_m[-1] < length_m:
                    distances_m.append(length_m)

            for order, dist_m in enumerate(distances_m):
                records_work.append(
                    {
                        "line_id": str(line_id),
                        "point_type": "equal_step",
                        "distance_m": float(dist_m),
                        "point_order": int(order),
                        "geometry": merged_line.interpolate(dist_m),
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

    def _collect_merged_lines(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> list[tuple[str, LineString]]:
        """
        Собирает линии в зависимости от source и превращает каждую группу
        в одну непрерывную трассу, по которой потом откладывается шаг.
        """
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
        """
        Группирует линии так, чтобы шаг шёл вдоль всей целевой трассы,
        а не сбрасывался на каждом сегменте.

        Базовый вариант:
        - MAIN_ONLY -> одна группа "main"
        - OTHER_ONLY -> одна группа "other"
        - ALL_LINES -> каждая feature отдельно, если нет лучшей бизнес-логики

        Если у тебя dataset уже содержит главную линию как одну feature,
        то MAIN_ONLY даст нужное поведение автоматически.
        """
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

        merged = linemerge(lines)
        return merged
