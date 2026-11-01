from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class GeoPoint:
    lat: float
    lon: float
    depth: Optional[float] = None

    def __post_init__(self) -> None:
        if not (-90 <= self.lat <= 90):
            raise ValueError(f"lat must be in [-90, 90], got {self.lat}")
        if not (-180 <= self.lon <= 180):
            raise ValueError(f"lon must be in [-180, 180], got {self.lon}")


@dataclass(frozen=True, slots=True)
class GeoLine:
    start: GeoPoint
    end: GeoPoint


@dataclass
class BathymetryGrid:
    lats: np.ndarray
    lons: np.ndarray
    z: np.ndarray
    source: str = "unknown"
    resolution_arcsec: float = 15.0

    def __post_init__(self) -> None:
        if self.lats.ndim != 1:
            raise ValueError("lats must be 1D")
        if self.lons.ndim != 1:
            raise ValueError("lons must be 1D")
        if self.z.ndim != 2:
            raise ValueError("z must be 2D")
        if self.z.shape != (len(self.lats), len(self.lons)):
            raise ValueError(
                f"Shape mismatch: z.shape={self.z.shape}, "
                f"expected=({len(self.lats)}, {len(self.lons)})"
            )

    @property
    def south(self) -> float:
        return float(self.lats.min())

    @property
    def north(self) -> float:
        return float(self.lats.max())

    @property
    def west(self) -> float:
        return float(self.lons.min())

    @property
    def east(self) -> float:
        return float(self.lons.max())

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape

    @property
    def min_depth(self) -> float:
        return float(np.nanmin(self.z))

    @property
    def max_depth(self) -> float:
        return float(np.nanmax(self.z))


@dataclass
class BathymetryProfile:
    line: GeoLine
    distances: np.ndarray
    depths: np.ndarray
    points: list[GeoPoint]

    def __post_init__(self) -> None:
        if len(self.distances) != len(self.depths):
            raise ValueError("distances and depths must have the same length")
        if len(self.distances) != len(self.points):
            raise ValueError("distances and points must have the same length")

    @property
    def total_length_m(self) -> float:
        return float(self.distances[-1]) if len(self.distances) > 0 else 0.0

    @property
    def min_depth(self) -> float:
        return float(np.nanmin(self.depths))

    @property
    def max_depth(self) -> float:
        return float(np.nanmax(self.depths))

    @property
    def num_points(self) -> int:
        return len(self.distances)
