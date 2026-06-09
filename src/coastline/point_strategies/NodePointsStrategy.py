from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point

from src.coastline.point_strategies.PointExtractionStrategy import (
    PointExtractionStrategy,
    PointSource,
)


class NodePointsStrategy(PointExtractionStrategy):
    """
    Извлекает узловые точки линий:
    - endpoints_only=False → все вершины линий;
    - endpoints_only=True  → только начало и конец каждой линии.

    unique=True убирает дубликаты с точностью до `precision` знаков.
    """

    def __init__(
        self,
        source: PointSource = PointSource.ALL_LINES,
        endpoints_only: bool = False,
        unique: bool = True,
        precision: int = 8,
    ) -> None:
        super().__init__(source)
        self.endpoints_only = endpoints_only
        self.unique = unique
        self.precision = precision

    def _extract_from_lines(self, gdf: gpd.GeoDataFrame) -> list[dict]:
        records: list[dict] = []
        seen: set[tuple[float, float]] = set()

        for line_id, line in self._iter_lines(gdf):
            coords = list(line.coords)
            if len(coords) < 2:
                continue

            pts = [coords[0], coords[-1]] if self.endpoints_only else coords

            for order, (x, y) in enumerate(pts):
                key = (round(x, self.precision), round(y, self.precision))
                if self.unique and key in seen:
                    continue
                seen.add(key)

                records.append(
                    {
                        "line_id": str(line_id),
                        "point_type": "node",
                        "point_order": order,
                        "geometry": Point(x, y),
                    }
                )

        return records
