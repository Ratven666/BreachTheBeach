from __future__ import annotations

import geopandas as gpd

from src.coastline.point_strategies.PointExtractionStrategy import (
    PointExtractionStrategy,
    PointSource,
)


class EqualRadiusStrategy(PointExtractionStrategy):
    """
    Точки с равным расстоянием (радиусом) от начала каждой линии:
    0, R, 2R, 3R, ... вдоль трассы.

    Семантически отличается от EqualStepAlongLineStrategy:
    шаг отсчитывается от старта, а не от предыдущей точки.
    Числовой результат идентичен, но намерение — задать
    зоны досягаемости / буферные радиусы от начальной точки.
    """

    def __init__(
        self,
        radius_step: float,
        source: PointSource = PointSource.ALL_LINES,
        include_origin: bool = True,
        include_endpoint: bool = True,
    ) -> None:
        super().__init__(source)
        if radius_step <= 0:
            raise ValueError("radius_step must be > 0")
        self.radius_step = float(radius_step)
        self.include_origin = include_origin
        self.include_endpoint = include_endpoint

    def _extract_from_lines(self, gdf: gpd.GeoDataFrame) -> list[dict]:
        records: list[dict] = []

        for line_id, line in self._iter_lines(gdf):
            length = float(line.length)
            if length == 0:
                continue

            start = 0.0 if self.include_origin else self.radius_step
            radii: list[float] = []
            r = start
            while r <= length:
                radii.append(r)
                r += self.radius_step

            if self.include_endpoint and (not radii or radii[-1] < length):
                radii.append(length)

            for order, radius in enumerate(radii):
                records.append(
                    {
                        "line_id": str(line_id),
                        "point_type": "equal_radius",
                        "radius": radius,
                        "point_order": order,
                        "geometry": line.interpolate(radius),
                    }
                )

        return records
