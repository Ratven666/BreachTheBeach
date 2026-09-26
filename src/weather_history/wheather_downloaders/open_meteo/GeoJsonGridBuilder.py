from __future__ import annotations

from math import ceil, floor
from pathlib import Path

import geopandas as gpd

from .models import GridPoint


class GeoJsonGridBuilder:
    def __init__(
        self,
        grid_step: float = 0.25,
        grid_center_offset: float = 0.125,
        cover_points_with_cells: bool = True,
        extra_border_cells: int = 1,
    ) -> None:
        self.grid_step = grid_step
        self.grid_center_offset = grid_center_offset
        self.cover_points_with_cells = cover_points_with_cells
        self.extra_border_cells = extra_border_cells

    def extract_points(self, vector_path: str | Path) -> list[tuple[float, float]]:
        path = Path(vector_path)

        last_error = None
        for encoding in (None, "utf-8", "utf-8-sig", "cp1251", "latin1"):
            try:
                kwargs = {}
                if encoding is not None:
                    kwargs["encoding"] = encoding
                gdf = gpd.read_file(path, **kwargs)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Failed to read vector file: {path}") from last_error

        if gdf.empty:
            raise ValueError(f"Vector file is empty: {path}")

        points: list[tuple[float, float]] = []

        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue

            geom_type = geom.geom_type

            if geom_type == "Point":
                points.append((float(geom.x), float(geom.y)))

            elif geom_type == "MultiPoint":
                for part in geom.geoms:
                    points.append((float(part.x), float(part.y)))

            elif geom_type == "LineString":
                for x, y in geom.coords:
                    points.append((float(x), float(y)))

            elif geom_type == "MultiLineString":
                for line in geom.geoms:
                    for x, y in line.coords:
                        points.append((float(x), float(y)))

            elif geom_type == "Polygon":
                for x, y in geom.exterior.coords:
                    points.append((float(x), float(y)))
                for ring in geom.interiors:
                    for x, y in ring.coords:
                        points.append((float(x), float(y)))

            elif geom_type == "MultiPolygon":
                for polygon in geom.geoms:
                    for x, y in polygon.exterior.coords:
                        points.append((float(x), float(y)))
                    for ring in polygon.interiors:
                        for x, y in ring.coords:
                            points.append((float(x), float(y)))

            else:
                raise ValueError(f"Unsupported geometry type: {geom_type}")

        if not points:
            raise ValueError(f"No coordinates found in vector file: {path}")

        return points

    def bbox_from_points(self, points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    def build_axis_centers_covering_points(
        self,
        min_value: float,
        max_value: float,
    ) -> tuple[list[float], tuple[int, int]]:
        half_cell = self.grid_step / 2 if self.cover_points_with_cells else 0.0
        extra = self.extra_border_cells * self.grid_step

        min_center_value = min_value - half_cell - extra
        max_center_value = max_value + half_cell + extra

        start_idx = ceil((min_center_value - self.grid_center_offset) / self.grid_step)
        end_idx = floor((max_center_value - self.grid_center_offset) / self.grid_step)

        centers = [
            round(self.grid_center_offset + index * self.grid_step, 6)
            for index in range(start_idx, end_idx + 1)
        ]
        return centers, (start_idx, end_idx)

    def build_grid(
        self,
        vector_path: str | Path,
    ) -> tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        list[GridPoint],
    ]:
        points = self.extract_points(vector_path)
        source_bbox = self.bbox_from_points(points)

        min_lon, min_lat, max_lon, max_lat = source_bbox

        latitudes, (lat_start_idx, lat_end_idx) = self.build_axis_centers_covering_points(
            min_value=min_lat,
            max_value=max_lat,
        )
        longitudes, (lon_start_idx, lon_end_idx) = self.build_axis_centers_covering_points(
            min_value=min_lon,
            max_value=max_lon,
        )

        grid: list[GridPoint] = []
        for lat_index, lat in enumerate(latitudes):
            for lon_index, lon in enumerate(longitudes):
                ring_y = 0
                ring_x = 0

                if lat_index == 0:
                    ring_y = -self.extra_border_cells
                elif lat_index == len(latitudes) - 1:
                    ring_y = self.extra_border_cells

                if lon_index == 0:
                    ring_x = -self.extra_border_cells
                elif lon_index == len(longitudes) - 1:
                    ring_x = self.extra_border_cells

                grid.append(
                    GridPoint(
                        lat=lat,
                        lon=lon,
                        ring_y=ring_y,
                        ring_x=ring_x,
                    )
                )

        weather_bbox = (
            min(longitudes) - self.grid_step / 2,
            min(latitudes) - self.grid_step / 2,
            max(longitudes) + self.grid_step / 2,
            max(latitudes) + self.grid_step / 2,
        )

        return source_bbox, weather_bbox, grid
