from __future__ import annotations

import numpy as np
from pyproj import Geod

from src.bathymetry.domain.models import BathymetryGrid, BathymetryProfile, GeoLine, GeoPoint
from src.bathymetry.interpolation.DepthInterpolator import DepthInterpolator

_WGS84 = Geod(ellps="WGS84")


class ProfileBuilder:
    def __init__(
        self,
        grid: BathymetryGrid,
        n_points: int = 200,
        interp_method: str = "linear",
    ) -> None:
        if n_points < 2:
            raise ValueError("n_points must be >= 2")
        self._grid = grid
        self._n_points = n_points
        self._interpolator = DepthInterpolator(grid, method=interp_method)

    def build(self, line: GeoLine) -> BathymetryProfile:
        points = self._sample_points(line, self._n_points)
        distances = self._compute_distances(line.start, points)
        depths = self._interpolator.depths_at_points(points)

        points_with_depth = [
            GeoPoint(lat=p.lat, lon=p.lon, depth=float(depth))
            for p, depth in zip(points, depths)
        ]

        return BathymetryProfile(
            line=line,
            distances=distances,
            depths=depths,
            points=points_with_depth,
        )

    @staticmethod
    def _sample_points(line: GeoLine, n: int) -> list[GeoPoint]:
        if n == 2:
            return [line.start, line.end]

        middle_points = _WGS84.npts(
            line.start.lon,
            line.start.lat,
            line.end.lon,
            line.end.lat,
            npts=n - 2,
        )

        all_points = [
            (line.start.lon, line.start.lat),
            *middle_points,
            (line.end.lon, line.end.lat),
        ]
        return [GeoPoint(lat=lat, lon=lon) for lon, lat in all_points]

    @staticmethod
    def _compute_distances(origin: GeoPoint, points: list[GeoPoint]) -> np.ndarray:
        distances = np.zeros(len(points), dtype=np.float64)
        for i, point in enumerate(points):
            _, _, dist = _WGS84.inv(origin.lon, origin.lat, point.lon, point.lat)
            distances[i] = dist
        return distances
