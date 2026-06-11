from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from src.base.BBox import BBox
from src.bathymetry.cache.bathymetry_cache import BathymetryCache
from src.bathymetry.domain.models import BathymetryGrid, BathymetryProfile, GeoLine, GeoPoint
from src.bathymetry.errors import BathymetryLoadError, BathymetryNotLoadedError
from src.bathymetry.exporters.base import BathymetryExportStrategy, ProfileExportStrategy
from src.bathymetry.interpolation.depth_interpolator import DepthInterpolator
from src.bathymetry.loaders.base import BathymetryLoader
from src.bathymetry.profile.profile_builder import ProfileBuilder
from src.bathymetry.visualization.grid_plotter import BathymetryGridPlotter
from src.bathymetry.visualization.profile_plotter import ProfilePlotter


class BathymetryService:
    def __init__(
        self,
        loader: BathymetryLoader,
        cache: Optional[BathymetryCache] = None,
        fallback_loader: BathymetryLoader | None = None,
        n_profile_points: int = 200,
        interp_method: str = "linear",
    ) -> None:
        self._loader = loader
        self._cache = cache
        self._fallback_loader = fallback_loader
        self._n_profile_points = n_profile_points
        self._interp_method = interp_method
        self._log = logger.bind(cls=self.__class__.__name__)

        self._grid: Optional[BathymetryGrid] = None
        self._interpolator: Optional[DepthInterpolator] = None

    def fetch(self, bbox: BBox, *, force_reload: bool = False) -> BathymetryGrid:
        primary_source = self._loader.source_name

        if not force_reload and self._cache and self._cache.has(bbox, primary_source):
            self._log.info("Loading bathymetry from cache")
            grid = self._cache.load(bbox, primary_source)
            self._set_current_grid(grid)
            return grid

        try:
            self._log.info(f"Loading bathymetry from source: {self._loader.source_name}")
            grid = self._loader.load(bbox)
        except BathymetryLoadError as e:
            if self._fallback_loader is None:
                raise
            self._log.warning(f"Primary loader failed: {e}")
            self._log.info(
                f"Loading bathymetry from fallback source: {self._fallback_loader.source_name}"
            )
            grid = self._fallback_loader.load(bbox)

        if self._cache:
            self._cache.save(bbox, grid)

        self._set_current_grid(grid)
        return grid

    def depth_at_point(self, point: GeoPoint) -> float:
        self._require_grid()
        return self._interpolator.depth_at(point)

    def build_profile(
        self,
        line: GeoLine,
        *,
        n_points: Optional[int] = None,
        interp_method: Optional[str] = None,
    ) -> BathymetryProfile:
        self._require_grid()

        builder = ProfileBuilder(
            grid=self._grid,
            n_points=n_points or self._n_profile_points,
            interp_method=interp_method or self._interp_method,
        )
        return builder.build(line)

    def plot_grid(
        self,
        title: Optional[str] = None,
        contour_interval: Optional[float] = None,
        output_path: Optional[str | Path] = None,
        show: bool = True,
    ):
        self._require_grid()
        plotter = BathymetryGridPlotter()
        return plotter.plot(
            self._grid,
            title=title,
            contour_interval=contour_interval,
            output_path=output_path,
            show=show,
        )

    def plot_profile(
        self,
        profile: BathymetryProfile,
        title: Optional[str] = None,
        output_path: Optional[str | Path] = None,
        show: bool = True,
    ):
        plotter = ProfilePlotter()
        return plotter.plot(profile, title=title, output_path=output_path, show=show)

    def export_grid(
        self,
        strategy: BathymetryExportStrategy,
        output_path: str | Path,
    ) -> Path:
        self._require_grid()
        return strategy.export(self._grid, output_path)

    def export_profile(
        self,
        profile: BathymetryProfile,
        strategy: ProfileExportStrategy,
        output_path: str | Path,
    ) -> Path:
        return strategy.export(profile, output_path)

    @property
    def grid(self) -> Optional[BathymetryGrid]:
        return self._grid

    @property
    def is_loaded(self) -> bool:
        return self._grid is not None

    def _set_current_grid(self, grid: BathymetryGrid) -> None:
        self._grid = grid
        self._interpolator = DepthInterpolator(grid, method=self._interp_method)

    def _require_grid(self) -> None:
        if self._grid is None or self._interpolator is None:
            raise BathymetryNotLoadedError(
                "Bathymetry grid is not loaded. Call fetch(bbox) first."
            )
