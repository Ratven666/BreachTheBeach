from __future__ import annotations

import geopandas as gpd

from src.coastline.point_strategies.PointExtractionStrategy import (
    PointExtractionStrategy,
    PointSource,
)


class EqualStepAlongLineStrategy(PointExtractionStrategy):
    """
    Точки через равный шаг `step` вдоль каждой линии.
    Использует shapely.interpolate(distance).
    """

    def __init__(
        self,
        step: float,
        source: PointSource = PointSource.ALL_LINES,
        include_endpoints: bool = True,
    ) -> None:
        super().__init__(source)
        if step <= 0:
            raise ValueError("step must be > 0")
        self.step = float(step)
        self.include_endpoints = include_endpoints

    def _extract_from_lines(self, gdf: gpd.GeoDataFrame) -> list[dict]:
        records: list[dict] = []

        for line_id, line in self._iter_lines(gdf):
            length = float(line.length)
            if length == 0:
                continue

            distances: list[float] = []
            d = 0.0
            while d <= length:
                distances.append(d)
                d += self.step

            if self.include_endpoints and distances[-1] < length:
                distances.append(length)

            for order, dist in enumerate(distances):
                records.append(
                    {
                        "line_id": str(line_id),
                        "point_type": "equal_step",
                        "distance": dist,
                        "point_order": order,
                        "geometry": line.interpolate(dist),
                    }
                )

        return records
