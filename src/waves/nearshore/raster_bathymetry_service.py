from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.sample import sample_gen

from bathymetry import BathymetryProfile


@dataclass
class RasterBathymetryService:
    """Привязан к конкретным (origin_lon, origin_lat) при создании."""
    raster_path: Path
    origin_lon: float   # ← координаты ЭТОЙ точки, заданы при __init__
    origin_lat: float
    radius_m: float = 20_000.0
    n_steps: int = 200
    band: int = 1
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def get_profile(self, direction: int) -> BathymetryProfile:
        """Строит трансект из origin_lon/lat в направлении direction,
        семплирует GeoTIFF, возвращает BathymetryProfile(depths_m=...)."""
        direction = int(direction) % 360
        if direction not in self._cache:
            self._cache[direction] = self._sample_transect(direction)
        return self._cache[direction]

    def _sample_transect(self, direction: int) -> BathymetryProfile:
        distances = np.linspace(0.0, self.radius_m, self.n_steps)
        lons, lats = self._build_transect_coords(direction, distances)
        depths = self._read_depths(lons, lats)  # rasterio + CRS reproject
        return BathymetryProfile(direction=direction,
                                 depths_m=depths, distances_m=distances)

    def _read_depths(self, lons, lats) -> np.ndarray:
        with rasterio.open(self.raster_path) as src:
            # перепроецируем если CRS растра не WGS-84
            if src.crs.to_epsg() != 4326:
                xs, ys = Transformer.from_crs("EPSG:4326", src.crs,
                                               always_xy=True).transform(lons, lats)
            else:
                xs, ys = lons, lats
            raw = np.array([float(s[0]) for s in sample_gen(src, zip(xs, ys))])
        # GEBCO: elevation < 0 → глубина > 0; суша → NaN
        depths = np.where(raw < 0.0, -raw, np.nan)
        if src.nodata is not None:
            depths = np.where(np.isclose(raw, src.nodata), np.nan, depths)
        return depths


@dataclass
class PointAwareBathymetryFactory:
    """Фабрика: создаёт per-point RasterBathymetryService."""
    raster_path: Path
    radius_m: float = 20_000.0
    n_steps: int = 200
    band: int = 1

    def for_point(self, lon: float, lat: float) -> RasterBathymetryService:
        """Вызывается батч-циклом для каждой точки перед WaveClimateService."""
        return RasterBathymetryService(
            raster_path=self.raster_path,
            origin_lon=lon, origin_lat=lat,
            radius_m=self.radius_m, n_steps=self.n_steps, band=self.band,
        )
