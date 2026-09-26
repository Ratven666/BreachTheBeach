from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.bathymetry.domain.models import BathymetryGrid, GeoPoint


class DepthInterpolator:
    def __init__(self, grid: BathymetryGrid, method: str = "linear") -> None:
        self._grid = grid
        self._method = method
        self._interp = self._build(grid, method)

    def depth_at(self, point: GeoPoint) -> float:
        self._validate_bounds(point)
        result = self._interp([[point.lat, point.lon]])
        return float(result[0])

    def depths_at_points(self, points: list[GeoPoint]) -> np.ndarray:
        for point in points:
            self._validate_bounds(point)
        coords = np.array([[p.lat, p.lon] for p in points], dtype=np.float64)
        return self._interp(coords).astype(np.float64)

    def change_method(self, method: str) -> "DepthInterpolator":
        return DepthInterpolator(self._grid, method=method)

    @staticmethod
    def _build(grid: BathymetryGrid, method: str) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            points=(grid.lats, grid.lons),
            values=grid.z,
            method=method,
            bounds_error=True,
            fill_value=np.nan,
        )

    def _validate_bounds(self, point: GeoPoint) -> None:
        g = self._grid
        if not (g.south <= point.lat <= g.north):
            raise ValueError(f"lat={point.lat} is outside grid bounds [{g.south}, {g.north}]")
        if not (g.west <= point.lon <= g.east):
            raise ValueError(f"lon={point.lon} is outside grid bounds [{g.west}, {g.east}]")
